"""Análisis automático, validación explícita, modelos versionados. Sin correcciones silenciosas."""
from pathlib import Path
from datetime import datetime, timezone
from io import BytesIO
import json, uuid, unicodedata
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA=ROOT/'data'
MODELS=ROOT/'models'
VERSIONS=MODELS/'versiones.json'
ACTIVE=MODELS/'activo.json'
SCHEMA=MODELS/'esquema.json'
MONTHS=['Enero','Febrero','Marzo','Abril','Mayo','Junio','Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre']
FEATURES=['municipio','mes_num','mes_sin','mes_cos','anio']
LABELS=['Baja','Media','Alta']

def norm(v):
    return ''.join(c for c in unicodedata.normalize('NFKD',str(v).strip().upper()) if not unicodedata.combining(c))

def data_json(v):
    if isinstance(v,(np.integer,)):return int(v)
    if isinstance(v,(np.floating,)):return None if not np.isfinite(v) else float(v)
    if isinstance(v,(pd.Timestamp,datetime)):return v.isoformat()
    if isinstance(v,(np.bool_,)):return bool(v)
    if pd.isna(v):return None
    return v

def clean_records(df):
    return json.loads(df.to_json(orient='records',date_format='iso',force_ascii=False))

def load_upload(blob,name):
    if not blob:raise ValueError('El archivo está vacío.')
    if len(blob)>15*1024*1024:raise ValueError('Archivo superior a 15 MB.')
    ext=Path(name).suffix.lower()
    if ext=='.csv':
        for encoding in ('utf-8-sig','latin-1'):
            try:
                df=pd.read_csv(BytesIO(blob),encoding=encoding,sep=None,engine='python')
                break
            except UnicodeDecodeError:continue
    elif ext in ('.xlsx','.xls'):
        df=pd.read_excel(BytesIO(blob))
    else:raise ValueError('Formato no permitido. Usa .csv o .xlsx.')
    if len(df)>100_000:raise ValueError('Máximo 100.000 filas.')
    if df.empty:raise ValueError('El archivo no contiene registros.')
    if df.columns.duplicated().any():raise ValueError('Columnas repetidas: corrige el archivo.')
    return df

def schema():return json.loads(SCHEMA.read_text(encoding='utf-8'))
def versions():return json.loads(VERSIONS.read_text(encoding='utf-8'))
def active_id():return json.loads(ACTIVE.read_text(encoding='utf-8'))['version']
def get_version(version=None):
    version=version or active_id()
    return next((x for x in versions() if x['version']==version),None)

def validate(df,labeled=False):
    s=schema()
    required=list(s['required_prediction'])+(s['required_training'] if labeled else [])
    errs=[]; warnings=[]
    missing=[c for c in required if c not in df.columns]
    for c in missing:errs.append({'variable':c,'problema':'Columna obligatoria ausente','esperado':s['fields'][c]['type'],'recibido':'No existe','filas_afectadas':len(df)})
    extra=[c for c in df if c not in s['fields']]
    if extra:warnings.append({'variable':', '.join(extra),'problema':'Columnas adicionales ignoradas por el modelo','filas_afectadas':len(df)})
    for col in required:
        if col not in df:continue
        x=df[col]
        missing_values=x.isna() | x.astype(str).str.strip().eq('')
        if missing_values.any():errs.append({'variable':col,'problema':'Valores nulos o vacíos','esperado':'No se permiten faltantes','recibido':'Faltantes','filas_afectadas':int(missing_values.sum()),'filas_ejemplo':(np.flatnonzero(missing_values)[:5]+2).tolist()})
        valid=x[~missing_values]
        if col=='municipio':
            normalized=valid.astype(str).map(norm)
            blank=normalized.eq('')
            if blank.any():errs.append({'variable':col,'problema':'Municipio vacío','esperado':'Nombre de municipio','recibido':'Vacío','filas_afectadas':int(blank.sum())})
            known=set(get_version()['municipios_aptos'])
            unknown=~normalized.isin(known)
            if unknown.any():errs.append({'variable':col,'problema':'Municipio sin modelo apto (historial insuficiente o desconocido)','esperado':f'Municipio apto del modelo activo ({len(known)} disponibles)','recibido':', '.join(sorted(normalized[unknown].unique())[:8]),'filas_afectadas':int(unknown.sum())}) if not labeled else warnings.append({'variable':col,'problema':'Municipio sin historial apto; candidato podrá incluirlo si acumula datos suficientes','filas_afectadas':int(unknown.sum())})
        elif col in ('anio','mes_num','visitantes'):
            numeric=pd.to_numeric(valid,errors='coerce')
            bad=numeric.isna()|(~np.isfinite(numeric))
            if bad.any():errs.append({'variable':col,'problema':'Tipo numérico inválido','esperado':'Número entero','recibido':', '.join(valid[bad].astype(str).unique()[:5]),'filas_afectadas':int(bad.sum())})
            numeric=numeric[~bad]
            if col=='anio':invalid=(numeric%1!=0)|(numeric<2020)|(numeric>2100)
            elif col=='mes_num':invalid=(numeric%1!=0)|(numeric<1)|(numeric>12)
            else:invalid=(numeric%1!=0)|(numeric<0)
            if invalid.any():errs.append({'variable':col,'problema':'Valor fuera de dominio','esperado':s['fields'][col]['rule'],'recibido':', '.join(map(str,numeric[invalid].unique()[:5])),'filas_afectadas':int(invalid.sum())})
    if {'municipio','anio','mes_num'}.issubset(df.columns) and not missing:
        duplicates=df.duplicated(['municipio','anio','mes_num'],keep=False)
        if duplicates.any():
            errs.append({'variable':'municipio, anio, mes_num','problema':'Periodo municipal duplicado: la predicción requiere un registro por municipio y mes','esperado':'Combinación única','recibido':'Duplicado','filas_afectadas':int(duplicates.sum())})
    state='Incompatible' if errs else 'Compatible con advertencias' if warnings else 'Compatible'
    return {'estado':state,'compatible':not bool(errs),'errores':errs,'advertencias':warnings,'columnas_esperadas':required,'columnas_recibidas':list(df.columns),'filas':len(df),'version_modelo':active_id()}

def features(df):
    x=df[['municipio','anio','mes_num']].copy()
    x['municipio']=x.municipio.map(norm)
    x['mes_sin']=np.sin(2*np.pi*x.mes_num/12)
    x['mes_cos']=np.cos(2*np.pi*x.mes_num/12)
    return x[FEATURES]

def cuts(train):
    q=train.visitantes.quantile([1/3,2/3]).values.astype(float)
    return [float(q[0]),float(q[1])] if np.isfinite(q).all() and q[0]<q[1] else None

def aptos(train):
    group=train.groupby('municipio')
    return {muni:cuts(g) for muni,g in group if len(g)>=24 and g.anio.nunique()>=3 and cuts(g) is not None}

def fit(train):
    eligible=aptos(train)
    subset=train[train.municipio.isin(eligible)]
    if subset.empty:raise ValueError('Sin suficientes registros para entrenar.')
    prep=ColumnTransformer([('categoricas',OneHotEncoder(handle_unknown='ignore'),['municipio','mes_num']),('continuas','passthrough',['mes_sin','mes_cos','anio'])])
    model=Pipeline([('transformar',prep),('bosque',RandomForestRegressor(n_estimators=240,min_samples_leaf=2,max_features=.85,random_state=42,n_jobs=-1))])
    model.fit(features(subset),np.log1p(subset.visitantes))
    return model, eligible

def level(n,t):return 'Baja' if n<=t[0] else 'Media' if n<=t[1] else 'Alta'

def bundle(version=None):
    item=get_version(version)
    if item is None:raise ValueError('Versión no encontrada.')
    return joblib.load(MODELS/item['archivo']),item

def predict(df,version=None):
    report=validate(df)
    if not report['compatible']:raise ValueError('Archivo incompatible. Consulta /api/validar.')
    model,meta=bundle(version)
    enriched=df.copy()
    enriched['municipio']=enriched.municipio.map(norm)
    estimate=np.maximum(0,np.expm1(model.predict(features(enriched))))
    result=df.reset_index(drop=True).copy()
    result.insert(0,'registro',np.arange(1,len(df)+1))
    result['visitantes_estimados']=np.round(estimate,2)
    result['demanda_prevista']=[level(v,meta['umbrales'][m]) for v,m in zip(estimate,enriched.municipio)]
    result['version_modelo']=meta['version']
    summary=result.demanda_prevista.value_counts().reindex(LABELS,fill_value=0).to_dict()
    return {'filas':len(result),'resumen':summary,'resultados':clean_records(result),'version_modelo':meta['version'],'nota':'Estimaciones orientativas sobre visitantes extranjeros no residentes; no equivalen a turismo total.'}

def eda(df):
    kind={}; missing=[]; descriptive={};freq={};outliers={};charts=[]
    for col in df:
        s=df[col]; n=s.isna().sum()
        missing.append({'variable':col,'faltantes':int(n),'porcentaje':round(100*n/len(df),2)})
        if pd.api.types.is_datetime64_any_dtype(s):t='temporal'
        elif pd.api.types.is_numeric_dtype(s):t='numerica'
        elif any(k in col.lower() for k in ('fecha','date')) and pd.to_datetime(s,errors='coerce').notna().mean()>.8:t='temporal'
        else:t='categorica'
        kind[col]=t
        if t=='numerica':
            num=pd.to_numeric(s,errors='coerce').dropna()
            if len(num):
                descriptive[col]={k:round(float(v),3) for k,v in {'media':num.mean(),'mediana':num.median(),'min':num.min(),'max':num.max(),'desviacion':num.std(ddof=0)}.items()}
                q1,q3=num.quantile([.25,.75]);iqr=q3-q1
                outliers[col]={'criterio':'Tukey 1.5 × RIQ','cantidad':int(((num<q1-1.5*iqr)|(num>q3+1.5*iqr)).sum())} if iqr>0 else {'criterio':'RIQ=0, no se aplica regla','cantidad':0}
                counts,edges=np.histogram(num,bins=min(12,max(1,int(np.sqrt(len(num)))))) if num.nunique()>1 else (np.array([len(num)]),np.array([num.iloc[0],num.iloc[0]+1]))
                charts.append({'variable':col,'tipo':'histograma','labels':[f'{edges[i]:.1f}–{edges[i+1]:.1f}' for i in range(len(counts))],'values':counts.tolist()})
        elif t=='categorica':
            counts=s.fillna('(faltante)').astype(str).value_counts().head(12)
            freq[col]=[{'categoria':str(k),'cantidad':int(v),'porcentaje':round(100*v/len(df),2)} for k,v in counts.items()]
            charts.append({'variable':col,'tipo':'barras','labels':list(counts.index),'values':counts.tolist()})
        elif t=='temporal':
            date=pd.to_datetime(s,errors='coerce').dropna()
            if len(date):
                ct=date.dt.to_period('M').astype(str).value_counts().sort_index().tail(24)
                charts.append({'variable':col,'tipo':'temporal','labels':list(ct.index),'values':ct.to_list()})
    number=df.select_dtypes(include='number')
    relations=[]
    if number.shape[1]>1:
        corr=number.corr(numeric_only=True)
        for i,a in enumerate(corr.columns):
            for b in corr.columns[i+1:]:
                v=corr.loc[a,b]
                if pd.notna(v):relations.append({'variables':[a,b],'correlacion_pearson':round(float(v),3)})
    relations.sort(key=lambda x:abs(x['correlacion_pearson']),reverse=True)
    return {'filas':len(df),'columnas':len(df.columns),'tipos':kind,'faltantes':missing,'duplicados':int(df.duplicated().sum()),'estadisticas_numericas':descriptive,'frecuencias_categoricas':freq,'valores_atipicos':outliers,'relaciones':relations[:15],'graficos':charts[:30],'observacion':'Correlación no implica causalidad. Posibles atípicos se detectan con criterio de Tukey (1,5 × RIQ).'}

def initialize():
    MODELS.mkdir(exist_ok=True)
    if VERSIONS.exists() and ACTIVE.exists() and SCHEMA.exists():return
    history=pd.read_csv(DATA/'historico_municipal_mensual.csv')
    model,thresholds=fit(history)
    version='v1.0'
    joblib.dump(model,MODELS/'modelo_v1.0.joblib')
    comparison=pd.read_csv(DATA/'comparacion_modelos.csv')
    previous=comparison[(comparison.anio_prueba==2025)&(comparison.modelo=='Random Forest')]
    metric=float(previous.exactitud.iloc[0]) if len(previous) else None
    metadata={'version':version,'fecha_entrenamiento':datetime.now(timezone.utc).isoformat(),'registros_entrenamiento':len(history),'variables':FEATURES,'algoritmo':'RandomForestRegressor','configuracion':{'n_estimators':240,'min_samples_leaf':2,'max_features':.85,'random_state':42},'metrica_validacion':{'anio':2025,'exactitud':metric},'archivo':'modelo_v1.0.joblib','estado':'activo','municipios_aptos':sorted(thresholds),'umbrales':thresholds,'limite_historico':int(history.anio.max())}
    VERSIONS.write_text(json.dumps([metadata],indent=2,ensure_ascii=False),encoding='utf8')
    ACTIVE.write_text(json.dumps({'version':version}),encoding='utf8')
    s={'required_prediction':['municipio','anio','mes_num'],'required_training':['visitantes'],'fields':{'municipio':{'type':'texto','nullable':False,'rule':'Municipio apto conocido en la versión activa'},'anio':{'type':'entero','nullable':False,'rule':'2020 a 2100'},'mes_num':{'type':'entero','nullable':False,'rule':'1 a 12'},'visitantes':{'type':'entero','nullable':False,'rule':'Entero >= 0 (solo para entrenamiento)'}},'target':{'name':'visitantes','type':'entero'},'exclude':['registro','fecha','visitantes'],'unknown_categories':'Bloqueo de municipios sin modelo apto para predecir','min_months':24,'min_years':3,'model_version':version}
    SCHEMA.write_text(json.dumps(s,indent=2,ensure_ascii=False),encoding='utf8')

def retrain(df):
    check=validate(df,labeled=True)
    if not check['compatible']:raise ValueError('Datos etiquetados incompatibles. Consulta /api/validar?etiquetados=true.')
    old=pd.read_csv(DATA/'historico_municipal_mensual.csv')[['municipio','anio','mes_num','visitantes']]
    new=df[['municipio','anio','mes_num','visitantes']].copy()
    new['municipio']=new.municipio.map(norm)
    new[['anio','mes_num','visitantes']]=new[['anio','mes_num','visitantes']].astype(int)
    if len(new)<12:raise ValueError('Se requieren al menos 12 registros nuevos etiquetados para una comparación temporal mínima.')
    maxold=(old.anio.astype(int)*12+old.mes_num.astype(int)).max()
    if ((new.anio*12+new.mes_num)<=maxold).any():raise ValueError('Los registros nuevos deben ser posteriores al último mes del histórico original (2025).')
    if pd.concat([old,new]).duplicated(['municipio','anio','mes_num']).any():raise ValueError('Se detectaron registros municipales duplicados en la unión.')
    chronological=new.sort_values(['anio','mes_num','municipio']).reset_index(drop=True)
    # Holdout de período(s) finales completos; no se usan en entrenar el candidato.
    periods=sorted(set(zip(chronological.anio,chronological.mes_num)))
    if len(periods)<3:raise ValueError('Se requieren al menos tres meses diferentes en los nuevos datos para validación temporal.')
    holdout_periods=set(periods[max(1,int(len(periods)*.8)):]); mask=chronological.apply(lambda r:(r.anio,r.mes_num) in holdout_periods,axis=1)
    holdout=chronological[mask].copy(); addition=chronological[~mask].copy()
    if holdout.empty or addition.empty:raise ValueError('No se pudo separar entrenamiento y evaluación cronológicamente.')
    train=pd.concat([old,addition],ignore_index=True)
    candidate,thresholds=fit(train)
    active,active_meta=bundle()
    test=holdout[holdout.municipio.isin(set(thresholds)&set(active_meta['municipios_aptos']))]
    if len(test)<3:raise ValueError('El conjunto reservado debe contener al menos 3 registros de municipios aptos en ambas versiones.')
    estimate_candidate=np.maximum(0,np.expm1(candidate.predict(features(test))))
    estimate_active=np.maximum(0,np.expm1(active.predict(features(test))))
    actual=test.visitantes.to_numpy()
    score={'registros_prueba':len(test),'periodos_reservados':sorted([f'{a}-{b:02d}' for a,b in holdout_periods]),'mae_activo':round(float(mean_absolute_error(actual,estimate_active)),3),'mae_candidato':round(float(mean_absolute_error(actual,estimate_candidate)),3)}
    # Entrenar el candidato definitivo con TODOS los datos etiquetados (tras comparar en holdout).
    final,all_thresholds=fit(pd.concat([old,new],ignore_index=True))
    entries=versions();version=f'v1.{len(entries)}';filename=f'modelo_{version}.joblib'
    joblib.dump(final,MODELS/filename)
    metadata={'version':version,'fecha_entrenamiento':datetime.now(timezone.utc).isoformat(),'registros_entrenamiento':len(old)+len(new),'variables':FEATURES,'algoritmo':'RandomForestRegressor','configuracion':active_meta['configuracion'],'metrica_validacion':score,'archivo':filename,'estado':'candidato','municipios_aptos':sorted(all_thresholds),'umbrales':all_thresholds,'limite_historico':int(new.anio.max()),'comparado_con':active_id(),'decision':'Pendiente de aprobación manual'}
    entries.append(metadata);VERSIONS.write_text(json.dumps(entries,ensure_ascii=False,indent=2))
    (MODELS/f'datos_nuevos_{version}.csv').write_bytes(new.to_csv(index=False).encode('utf-8-sig'))
    (MODELS/f'reporte_{version}.json').write_text(json.dumps({'version':version,'comparacion':score,'decision':'Pendiente de aprobación manual','datos_nuevos':len(new),'nuevo_eda':eda(new),'conjunto_actualizado_eda':eda(pd.concat([old,new],ignore_index=True))},ensure_ascii=False,indent=2))
    return {'version':version,'estado':'candidato','comparacion':score,'decision':'El modelo activo permanece sin cambios. Revisa los resultados antes de activar manualmente.'}

def activate(version):
    items=versions();selected=next((it for it in items if it['version']==version),None)
    if not selected or selected['estado']!='candidato':raise ValueError('Solo puede activarse una versión candidata.')
    for item in items:
        if item['estado']=='activo':item['estado']='historico'
        if item['version']==version:item['estado']='activo';item['decision']='Aprobada manualmente'
    VERSIONS.write_text(json.dumps(items,ensure_ascii=False,indent=2))
    ACTIVE.write_text(json.dumps({'version':version}))
    s=schema();s['model_version']=version;SCHEMA.write_text(json.dumps(s,ensure_ascii=False,indent=2))
    return {'version_activa':version,'resultado':'Activación manual realizada'}
