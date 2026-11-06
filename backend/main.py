from contextlib import asynccontextmanager
from pathlib import Path
from io import BytesIO
import json
import os
import pandas as pd
from fastapi import FastAPI,UploadFile,File,HTTPException,Query
from fastapi.responses import HTMLResponse,StreamingResponse,FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from . import core

@asynccontextmanager
async def lifespan(app):
    core.initialize()
    yield

app=FastAPI(title='Turismo Chocó | Análisis y Predicción',version='1.0.0',description='EDA independiente, compatibilidad estricta, Random Forest persistido, reportes y modelos candidatos.',lifespan=lifespan)
app.mount('/static',StaticFiles(directory=core.ROOT/'frontend'),name='static')
CACHE={}

@app.get('/',include_in_schema=False)
def home():return FileResponse(core.ROOT/'frontend/index.html')

async def upload(file:UploadFile):
    try:return core.load_upload(await file.read(),file.filename or '')
    except Exception as e:raise HTTPException(400,str(e))

@app.get('/api/modelo')
def modelo():
    v=core.get_version()
    return {k:val for k,val in v.items() if k!='umbrales'} | {'esquema':core.schema(),'versiones':[{k:it[k] for k in ['version','estado','fecha_entrenamiento','registros_entrenamiento','metrica_validacion']} for it in core.versions()]}

@app.get('/api/historico')
def historico():
    df=pd.read_csv(core.DATA/'historico_municipal_mensual.csv')
    return {'total':len(df),'municipios':sorted(df.municipio.unique().tolist()),'anios':sorted(df.anio.unique().tolist()),'serie':core.clean_records(df.groupby(['anio','mes_num'],as_index=False).visitantes.sum()),'cobertura':core.clean_records(df.groupby('municipio',as_index=False).agg(meses=('visitantes','size'),visitantes=('visitantes','sum')))}

@app.get('/api/pronostico')
def pronostico(municipio:str|None=None):
    data=pd.read_csv(core.DATA/'predicciones_municipales_2026.csv')
    if municipio:data=data[data.municipio==core.norm(municipio)]
    return {'resultados':core.clean_records(data),'anio':2026,'nota':'Pronóstico de referencia de 2026 generado por el proyecto original; nuevas predicciones por archivo emplean la versión activa.'}

@app.post('/api/eda')
async def eda(file:UploadFile=File(...)):
    df=await upload(file)
    out=core.eda(df)
    CACHE['eda']={'json':out,'nombre':file.filename}
    return out

@app.post('/api/validar')
async def validar(file:UploadFile=File(...),etiquetados:bool=False):
    df=await upload(file)
    report=core.validate(df,labeled=etiquetados)
    CACHE['compatibilidad']={'json':report,'nombre':file.filename}
    return report

@app.post('/api/predecir')
async def predecir(file:UploadFile=File(...)):
    df=await upload(file)
    report=core.validate(df)
    CACHE['compatibilidad']={'json':report,'nombre':file.filename}
    if not report['compatible']:raise HTTPException(422,detail={'mensaje':'Archivo incompatible. No se realizaron predicciones.','validacion':report})
    try:
        result=core.predict(df)
        CACHE['predicciones']={'json':result,'nombre':file.filename}
        return result
    except Exception as e:raise HTTPException(400,str(e))

@app.post('/api/reentrenar')
async def reentrenar(file:UploadFile=File(...)):
    if os.getenv('CHOCODATA_PUBLIC_DEMO', '0') == '1':
        raise HTTPException(403, 'Reentrenamiento deshabilitado en la demostración pública para proteger el modelo.')
    df=await upload(file)
    report=core.validate(df,labeled=True)
    CACHE['compatibilidad']={'json':report,'nombre':file.filename}
    if not report['compatible']:raise HTTPException(422,detail={'mensaje':'No se puede reentrenar con datos incompatibles.','validacion':report})
    try:
        result=core.retrain(df)
        CACHE['actualizacion']={'json':result,'nombre':file.filename}
        return result
    except Exception as e:raise HTTPException(400,str(e))

class Activation(BaseModel):
    version:str
    confirmar:bool=False

@app.post('/api/activar')
def activar(body:Activation):
    if os.getenv('CHOCODATA_PUBLIC_DEMO', '0') == '1':
        raise HTTPException(403, 'Activación de versiones deshabilitada en la demostración pública.')
    if not body.confirmar:raise HTTPException(400,'Se requiere confirmación explícita de la activación.')
    try:return core.activate(body.version)
    except Exception as e:raise HTTPException(400,str(e))

@app.get('/api/reportes/{tipo}')
def reportes(tipo:str,formato:str=Query('json',pattern='^(json|csv|xlsx)$')):
    if tipo not in CACHE:raise HTTPException(404,'Ejecuta primero el análisis correspondiente.')
    obj=CACHE[tipo]['json']
    if formato=='json':
        return StreamingResponse(BytesIO(json.dumps(obj,ensure_ascii=False,indent=2).encode()),media_type='application/json',headers={'Content-Disposition':f'attachment; filename="reporte_{tipo}.json"'})
    if tipo=='predicciones':df=pd.DataFrame(obj['resultados'])
    elif tipo=='compatibilidad':df=pd.DataFrame(obj['errores']+obj['advertencias'])
    elif tipo=='eda':df=pd.DataFrame(obj['faltantes'])
    else:df=pd.json_normalize(obj)
    b=BytesIO()
    if formato=='xlsx':
        with pd.ExcelWriter(b,engine='openpyxl') as w:
            df.to_excel(w,sheet_name='Resultados',index=False)
            pd.DataFrame([{'reporte':tipo,'archivo':CACHE[tipo]['nombre']}]).to_excel(w,sheet_name='Contexto',index=False)
        media='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    else:
        b.write(df.to_csv(index=False).encode('utf-8-sig'));media='text/csv'
    b.seek(0)
    return StreamingResponse(b,media_type=media,headers={'Content-Disposition':f'attachment; filename="reporte_{tipo}.{formato}"'})

@app.get('/api/reportes/actualizacion/{version}')
def reporte_version(version:str):
    path=core.MODELS/f'reporte_{version}.json'
    if not path.exists():raise HTTPException(404,'Reporte no encontrado.')
    return FileResponse(path,filename=path.name,media_type='application/json')

@app.get('/api/salud')
def salud():return {'estado':'ok','version_activa':core.active_id(),'modo_publico':os.getenv('CHOCODATA_PUBLIC_DEMO', '0')=='1'}
