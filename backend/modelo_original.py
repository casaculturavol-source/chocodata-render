"""Predicción municipal 2026 con Random Forest y comparación temporal con línea base.

Ejecutar: python prediccion_municipal.py
Datos: visitantes extranjeros no residentes 2020-2025; no equivalen a turismo total.
Los meses sin registro NO se completan con cero. El modelo no puede inferir
por sí solo si hubo ausencia de turistas o ausencia de reporte.
"""
from pathlib import Path
import unicodedata
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, f1_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'resultados_choco' / 'visitantes_eda_limpio.csv'
OUTPUT = ROOT / 'resultados_municipales'
MONTHS = {'ENE':1,'FEB':2,'MAR':3,'ABR':4,'MAY':5,'JUN':6,'JUL':7,'AGO':8,'SEP':9,'OCT':10,'NOV':11,'DIC':12}
MONTH_NAMES = dict(enumerate(['Enero','Febrero','Marzo','Abril','Mayo','Junio','Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre'],1))
MIN_MONTHS, MIN_YEARS = 24, 3
CLASSES = ['Baja','Media','Alta']
FEATURES = ['municipio','mes_num','mes_sin','mes_cos','anio']


def normalize(value):
    raw = unicodedata.normalize('NFKD',str(value).strip().upper())
    return ''.join(ch for ch in raw if not unicodedata.combining(ch))


def load_monthly(path=SOURCE):
    df = pd.read_csv(path,encoding='utf-8-sig')
    required = {'a_o','mes','municipio_estandar','visitantes'}
    if not required.issubset(df):
        raise ValueError(f'Faltan columnas {required - set(df.columns)}')
    df['anio'] = pd.to_numeric(df.a_o,errors='raise').astype(int)
    df['mes_num'] = df.mes.map(lambda s: MONTHS.get(normalize(s)[:3]))
    if df.mes_num.isna().any(): raise ValueError('Mes no reconocido')
    df['mes_num'] = df.mes_num.astype(int)
    df['municipio'] = df.municipio_estandar.map(normalize)
    df['visitantes'] = pd.to_numeric(df.visitantes,errors='raise')
    if df.visitantes.isna().any() or (df.visitantes < 0).any(): raise ValueError('Conteos inválidos')
    monthly = df.groupby(['municipio','anio','mes_num'],as_index=False)['visitantes'].sum()
    monthly['fecha'] = pd.to_datetime(dict(year=monthly.anio,month=monthly.mes_num,day=1))
    return monthly.sort_values(['municipio','fecha']).reset_index(drop=True)


def add_features(df):
    d = df.copy()
    d['mes_sin'] = np.sin(2 * np.pi * d.mes_num / 12)
    d['mes_cos'] = np.cos(2 * np.pi * d.mes_num / 12)
    return d


def thresholds(train):
    q = train.visitantes.quantile([1/3,2/3]).values.astype(float)
    return (float(q[0]),float(q[1])) if np.isfinite(q).all() and q[0] < q[1] else None


def level(value,cuts):
    if value <= cuts[0]: return 'Baja'
    if value <= cuts[1]: return 'Media'
    return 'Alta'


def eligible(df):
    if len(df) < MIN_MONTHS: return False,'Historial insuficiente (<24 meses observados)'
    if df.anio.nunique() < MIN_YEARS: return False,'Historial insuficiente (<3 años distintos)'
    if thresholds(df) is None: return False,'Sin separación suficiente entre categorías'
    return True,'Apto: modelo de aprendizaje automático orientativo'


def seasonal_predict(train, month):
    overall = float(train.visitantes.median())
    same = train.loc[train.mes_num == month,'visitantes']
    if same.empty: return overall
    weight = len(same) / (len(same) + 2)
    return float(weight * same.median() + (1-weight) * overall)


def fit_ml(train):
    """Aprendizaje supervisado conjunto; transforma objetivo log(1+visitas).

    Incluye solo municipios aptos ANTES de cada corte para no introducir fugas.
    """
    approved = [m for m,g in train.groupby('municipio') if eligible(g)[0]]
    selected = add_features(train.loc[train.municipio.isin(approved)])
    if selected.empty: return None
    preprocessing = ColumnTransformer([
        ('categoricas',OneHotEncoder(handle_unknown='ignore'),['municipio','mes_num']),
        ('continuas','passthrough',['mes_sin','mes_cos','anio'])
    ])
    model = Pipeline([
        ('transformar',preprocessing),
        ('bosque',RandomForestRegressor(n_estimators=240,min_samples_leaf=2,
                   max_features=0.85,random_state=42,n_jobs=-1))
    ])
    model.fit(selected[FEATURES],np.log1p(selected.visitantes))
    return model


def predict_ml(model, rows):
    if model is None: return np.full(len(rows),np.nan)
    return np.maximum(0,np.expm1(model.predict(add_features(rows)[FEATURES])))


def evaluate_split(monthly,train_end,test_year):
    """Corte temporal, umbrales calculados únicamente con entrenamiento."""
    past = monthly.loc[monthly.anio <= train_end]
    test = monthly.loc[monthly.anio == test_year]
    model = fit_ml(past)
    records = []
    for muni, dev in past.groupby('municipio'):
        if not eligible(dev)[0]: continue
        t = test[test.municipio == muni]
        if t.empty: continue
        cuts = thresholds(dev)
        estimates = predict_ml(model,t)
        for row,estimate in zip(t.itertuples(),estimates):
            baseline = seasonal_predict(dev,row.mes_num)
            actual = level(row.visitantes,cuts)
            records.append({'municipio':muni,'fecha':row.fecha.date().isoformat(),
               'observado':row.visitantes,'estimado':round(float(estimate),2),
               'estimado_baseline':round(float(baseline),2),
               'nivel_real':actual,'nivel_estimado':level(estimate,cuts),
               'nivel_baseline':level(baseline,cuts),
               'acierto':actual == level(estimate,cuts),
               'acierto_baseline':actual == level(baseline,cuts)})
    return pd.DataFrame(records,columns=['municipio','fecha','observado','estimado','estimado_baseline','nivel_real','nivel_estimado','nivel_baseline','acierto','acierto_baseline'])


def metrics(evaluation,year):
    if evaluation.empty: return []
    records=[]
    for name,pred_col,count_col in [('Random Forest','nivel_estimado','estimado'),('Mediana estacional','nivel_baseline','estimado_baseline')]:
        y = evaluation.nivel_real
        p = evaluation[pred_col]
        records.append({'anio_prueba':year,'modelo':name,'observaciones':len(evaluation),
                        'municipios':evaluation.municipio.nunique(),
                        'exactitud':round(float((y==p).mean()),4),
                        'f1_macro':round(float(f1_score(y,p,labels=CLASSES,average='macro',zero_division=0)),4),
                        'exactitud_equilibrada':round(float(balanced_accuracy_score(y,p)),4),
                        'mae_visitantes':round(float(mean_absolute_error(evaluation.observado,evaluation[count_col])),3)})
    return records


def generar_tablas_generales(forecast, year):
    columns = ['municipio','anio','mes_nombre','afluencia_prevista','visitantes_estimados','estado','metodo','referencia_estacional']
    general = forecast.sort_values(['municipio','mes'])[columns].rename(columns={
       'municipio':'Municipio','anio':'Año','mes_nombre':'Mes',
       'afluencia_prevista':'Demanda prevista','visitantes_estimados':'Visitantes extranjeros estimados',
       'estado':'Estado','metodo':'Método','referencia_estacional':'Estimación base histórica'})
    general.to_csv(OUTPUT / f'tabla_general_demanda_{year}.csv',index=False,encoding='utf-8-sig')
    matrix = forecast.pivot(index='municipio',columns='mes',values='afluencia_prevista').reindex(columns=range(1,13)).rename(columns=MONTH_NAMES)
    matrix.index.name='Municipio'
    matrix.to_csv(OUTPUT / f'matriz_temporadas_{year}.csv',encoding='utf-8-sig')
    return general,matrix


def consultar(predicciones,municipio,mes):
    result = predicciones.loc[(predicciones.municipio == normalize(municipio)) & (predicciones.mes == int(mes))]
    if result.empty: raise ValueError('Municipio/mes no encontrado')
    return result.iloc[0].to_dict()


def run(year=2026):
    if year <= 2025: raise ValueError('Se requiere un año posterior a 2025')
    OUTPUT.mkdir(exist_ok=True)
    monthly = load_monthly()
    monthly.to_csv(OUTPUT / 'historico_municipal_mensual.csv',index=False,encoding='utf-8-sig')
    # Dos pruebas estrictamente posteriores a su ventana de entrenamiento.
    val_2024 = evaluate_split(monthly,2023,2024)
    test_2025 = evaluate_split(monthly,2024,2025)
    evaluations = pd.concat([val_2024,test_2025],ignore_index=True)
    evaluations.to_csv(OUTPUT / 'evaluacion_temporal_2024_2025.csv',index=False,encoding='utf-8-sig')
    test_2025.to_csv(OUTPUT / 'evaluacion_temporal_2025.csv',index=False,encoding='utf-8-sig')
    comparison = pd.DataFrame(metrics(val_2024,2024)+metrics(test_2025,2025))
    comparison.to_csv(OUTPUT / 'comparacion_modelos.csv',index=False,encoding='utf-8-sig')
    # Entrenamiento final SOLO tras la prueba temporal, usando años hasta 2025.
    model = fit_ml(monthly)
    report, forecast = [],[]
    for muni, group in monthly.groupby('municipio',sort=True):
        ok,reason = eligible(group)
        cuts = thresholds(group) if ok else None
        report.append({'municipio':muni,'meses_observados':len(group),
            'anios_distintos':group.anio.nunique(),'visitantes_registrados':group.visitantes.sum(),
            'apto':ok,'motivo':reason,'umbral_baja_media':cuts[0] if cuts else np.nan,
            'umbral_media_alta':cuts[1] if cuts else np.nan})
        future = pd.DataFrame({'municipio':[muni]*12,'anio':[year]*12,'mes_num':list(range(1,13))})
        estimates = predict_ml(model,future) if ok else [np.nan]*12
        for mo,estimate in enumerate(estimates,1):
            baseline = seasonal_predict(group,mo) if ok else np.nan
            forecast.append({'municipio':muni,'anio':year,'mes':mo,'mes_nombre':MONTH_NAMES[mo],
                'visitantes_estimados':round(float(estimate),2) if ok else np.nan,
                'afluencia_prevista':level(estimate,cuts) if ok else 'No estimable',
                'estado':reason,'metodo':'Random Forest' if ok else 'No estimable',
                'referencia_estacional':round(baseline,2) if ok else np.nan,
                'umbral_baja_media':cuts[0] if cuts else np.nan,
                'umbral_media_alta':cuts[1] if cuts else np.nan})
    coverage, predictions = pd.DataFrame(report),pd.DataFrame(forecast)
    coverage.to_csv(OUTPUT/'cobertura_municipal.csv',index=False,encoding='utf-8-sig')
    predictions.to_csv(OUTPUT/f'predicciones_municipales_{year}.csv',index=False,encoding='utf-8-sig')
    general,matrix = generar_tablas_generales(predictions,year)
    summary={'municipios':len(coverage),'municipios_aptos':int(coverage.apto.sum()),
        'predicciones_estimables':int((predictions.afluencia_prevista!='No estimable').sum()),
        'evaluaciones_2025':len(test_2025),
        'exactitud_2025_ml':float(test_2025.acierto.mean()) if len(test_2025) else None,
        'exactitud_2025_baseline':float(test_2025.acierto_baseline.mean()) if len(test_2025) else None}
    print('Resumen:',summary)
    print('\nComparación temporal:\n',comparison.to_string(index=False))
    return monthly,coverage,predictions,test_2025,summary


if __name__ == '__main__': run()
