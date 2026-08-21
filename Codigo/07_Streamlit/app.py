import streamlit as st
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from informe_pdf import generar_informe_pdf
st.title("Scouting por Similitud — Real Racing Club de Santander")
st.write("Bienvenido a la herramienta de análisis y scouting basada en similitud estadística.")
# ---------- RUTAS BASE (relativas a la ubicación de este archivo) ----------
# ESTA ES LA VERSIÓN PARA TU CARPETA LOCAL DE TRABAJO (PFM\Código\07_Streamlit),
# que conserva las subcarpetas numeradas de Datos. NO subir esta versión al
# repo de GitHub — ahí va app.py (estructura plana), no app_LOCAL.py.
#   PFM/                                      (tu carpeta de trabajo)
#   ├── Código/07_Streamlit/app.py, informe_pdf.py   <- este archivo está aquí
#   ├── Datos/03_Feature_engineering/jugadores_features.csv
#   ├── Datos/04_Modelo/similares_top15.csv
#   ├── Datos/04_Modelo/jugadores_pca_componentes.csv
#   ├── Datos/01_Fuentes_Originales/datagaffer_equipos_limpios.csv
#   └── Datos/06_Fotos_PDF/*.jpg
BASE_DIR = Path(__file__).resolve().parent  # Código/07_Streamlit
RUTA_DATOS = BASE_DIR.parent.parent / "Datos"  # sube 2 niveles a la raíz de PFM, baja a Datos
# ---------- CARGA DE DATOS ----------
@st.cache_data
def cargar_datos():
    jugadores = pd.read_csv(RUTA_DATOS / "03_Feature_engineering" / "jugadores_features.csv", encoding='utf-8-sig')
    similares = pd.read_csv(RUTA_DATOS / "04_Modelo" / "similares_top15.csv", encoding='utf-8-sig')
    equipos = pd.read_csv(RUTA_DATOS / "01_Fuentes_Originales" / "datagaffer_equipos_limpios.csv", encoding='utf-8-sig')
    pca = pd.read_csv(RUTA_DATOS / "04_Modelo" / "jugadores_pca_componentes.csv", encoding='utf-8-sig')
    # Corregir formato numérico por si alguna columna de estilo viniera con coma decimal
    columnas_estilo = ['ppda', 'agix_index', 'nec_index', 'control_index',
                        'ORtg', 'DRtg', 'DGRtg', 'consistency',
                        'pace_index', 'dgr_index']
    for col in columnas_estilo:
        equipos[col] = equipos[col].astype(str).str.replace(',', '.').astype(float)
    return jugadores, similares, equipos, pca
df_jugadores, df_similares, df_equipos, df_pca = cargar_datos()
RUTA_FOTOS = RUTA_DATOS / "06_Fotos_PDF"
# ---------- COMPATIBILIDAD TÁCTICA (generalizada) ----------
# NOTA: desde la corrección del 08/08/2026, los 89 nombres de 'team' en
# datagaffer_equipos_limpios.csv coinciden EXACTO (carácter a carácter) con
# 'club_actual' en jugadores_features.csv. Ya no hace falta fuzzy matching
# (rapidfuzz) ni correcciones_manuales: comparación directa de texto.
COLUMNAS_ESTILO = ['ppda', 'agix_index', 'nec_index', 'control_index',
                    'ORtg', 'DRtg', 'DGRtg', 'consistency',
                    'pace_index', 'dgr_index']
# Escalado de las columnas de estilo antes de calcular cualquier coseno.
# Sin esto, las 5 columnas en escala 0-100 (agix_index, nec_index, control_index,
# pace_index, dgr_index) dominan el coseno frente a las 5 en escalas pequeñas
# (ppda, ORtg, DRtg, DGRtg, consistency), inflando la compatibilidad de
# prácticamente cualquier par de clubes (~0.95-0.99). Se ajusta sobre las 89
# filas completas de df_equipos, no sobre pares individuales.
scaler_estilo = StandardScaler()
df_equipos_escalado = df_equipos.copy()
df_equipos_escalado[COLUMNAS_ESTILO] = scaler_estilo.fit_transform(df_equipos[COLUMNAS_ESTILO])
def obtener_vector_estilo(club_actual):
    vector = df_equipos_escalado.loc[df_equipos_escalado['team'] == club_actual, COLUMNAS_ESTILO].values
    return vector[0] if len(vector) > 0 else None
def calcular_compatibilidad(club_origen, club_candidato):
    vector_origen = obtener_vector_estilo(club_origen)
    vector_candidato = obtener_vector_estilo(club_candidato)
    if vector_origen is None or vector_candidato is None:
        return None
    dot = np.dot(vector_origen, vector_candidato)
    norm = np.linalg.norm(vector_origen) * np.linalg.norm(vector_candidato)
    return round(dot / norm, 4) if norm != 0 else None
# ---------- SCORE DE CONSENSO ----------
def calcular_consenso(df):
    """
    Combina Similitud, xValue y Compatibilidad_táctica en un único score de consenso.
    Normalización min-max dentro del propio top-15 (no contra todo el dataset).
    Compatibilidad_táctica ausente (None) se rellena con la media de las conocidas
    dentro de ese mismo top-15 antes de normalizar (tratamiento neutro, no penalizado).
    Pesos iguales (1/3) entre las tres componentes.
    """
    df = df.copy()
    media_compat = df['Compatibilidad_táctica'].mean()
    compat_rellena = df['Compatibilidad_táctica'].fillna(media_compat)
    def normalizar(serie):
        minimo, maximo = serie.min(), serie.max()
        if maximo == minimo:
            return pd.Series(0.5, index=serie.index)
        return (serie - minimo) / (maximo - minimo)
    similitud_norm = normalizar(df['Similitud'])
    xvalue_norm = normalizar(df['xValue'])
    compat_norm = normalizar(compat_rellena)
    df['Consenso'] = ((similitud_norm + xvalue_norm + compat_norm) / 3).round(4)
    return df
# ---------- COMPARADOR DIRECTO ENTRE DOS JUGADORES ----------
def comparar_dos_jugadores(df_pca, df_jugadores, jugador_ref, jugador_obj):
    """
    Calcula la similitud coseno exacta entre dos jugadores concretos y su posición
    en el ranking completo del grupo posicional (no solo el top-15 precomputado).
    Útil para casos de validación retrospectiva donde el fichaje real queda fuera
    del top-15: permite saber si el fallo es "por poco" o total.
    """
    fila_ref = df_pca[df_pca['Player'] == jugador_ref]
    if fila_ref.empty:
        return None, f"{jugador_ref} no está en el dataset de modelo (portero excluido, o sin datos de minutos)."
    grupo = fila_ref.iloc[0]['grupo_posicional']
    fila_obj = df_pca[df_pca['Player'] == jugador_obj]
    if fila_obj.empty:
        return None, f"{jugador_obj} no está en el dataset de modelo (portero excluido, o sin datos de minutos)."
    if fila_obj.iloc[0]['grupo_posicional'] != grupo:
        return None, (f"Grupos posicionales distintos: {jugador_ref} es '{grupo}', "
                       f"{jugador_obj} es '{fila_obj.iloc[0]['grupo_posicional']}'. "
                       f"La similitud coseno solo se calcula dentro del mismo grupo, "
                       f"así que no son comparables por diseño del modelo.")
    grupo_df = df_pca[df_pca['grupo_posicional'] == grupo].copy()
    # Solo columnas PC sin NaN para este grupo (algunos grupos necesitan menos
    # componentes que otros, ver sección 5.6 de la memoria)
    pc_cols = [c for c in grupo_df.columns if c.startswith('PC')]
    pc_cols_validas = [c for c in pc_cols if grupo_df[c].notna().all()]
    X = grupo_df[pc_cols_validas].values
    vector_ref = grupo_df.loc[grupo_df['Player'] == jugador_ref, pc_cols_validas].values
    grupo_df['similitud'] = cosine_similarity(vector_ref, X)[0]
    ranking_df = (grupo_df[grupo_df['Player'] != jugador_ref]
                  .sort_values('similitud', ascending=False)
                  .reset_index(drop=True))
    ranking_df['ranking'] = ranking_df.index + 1
    fila_resultado = ranking_df[ranking_df['Player'] == jugador_obj].iloc[0]
    total = len(ranking_df)
    return {
        'similitud': round(float(fila_resultado['similitud']), 4),
        'ranking': int(fila_resultado['ranking']),
        'total_candidatos': total,
        'percentil': round(100 * (1 - fila_resultado['ranking'] / total), 1),
        'en_top15': fila_resultado['ranking'] <= 15,
        'grupo_posicional': grupo,
    }, None
# ---------- BUSCADOR DE JUGADOR ----------
st.divider()
st.header("Buscador de jugador de referencia")
jugadores_disponibles = sorted(df_similares['Player'].unique())
jugador_seleccionado = st.selectbox(
    "Selecciona un jugador de referencia:",
    options=jugadores_disponibles,
    index=None,
    placeholder="Escribe o selecciona un jugador..."
)
if jugador_seleccionado:
    fila_jugador = df_similares[df_similares['Player'] == jugador_seleccionado].iloc[0]
    st.write(f"Posición: **{fila_jugador['grupo_posicional']}**")
    club_referencia = df_jugadores.loc[df_jugadores['Player'] == jugador_seleccionado, 'club_actual']
    club_referencia = club_referencia.values[0] if len(club_referencia) > 0 else None
    if club_referencia is None:
        st.warning("No se ha encontrado el club actual del jugador de referencia — la compatibilidad táctica no estará disponible.")
    top15 = []
    for rank in range(1, 16):
        top15.append({
            'Ranking': rank,
            'Jugador': fila_jugador[f'similar_{rank}'],
            'Similitud': fila_jugador[f'score_{rank}']
        })
    df_top15 = pd.DataFrame(top15)
    df_top15 = df_top15.merge(
        df_jugadores[['Player', 'club_actual', 'valor_mercado', 'xValue', 'rendimiento_esperado']],
        left_on='Jugador', right_on='Player', how='left'
    ).drop(columns=['Player'])
    if club_referencia:
        df_top15 = df_top15[df_top15['club_actual'] != club_referencia]
    df_top15['Compatibilidad_táctica'] = df_top15['club_actual'].apply(
        lambda club_candidato: calcular_compatibilidad(club_referencia, club_candidato)
    )
    df_top15 = calcular_consenso(df_top15)
    st.subheader(f"Top 15 jugadores similares a {jugador_seleccionado}")
    if club_referencia:
        st.caption(f"La compatibilidad táctica se calcula respecto al estilo de juego de **{club_referencia}** (club actual de {jugador_seleccionado})")
    columna_orden = st.selectbox(
        "Ordenar por:",
        options=['Similitud', 'xValue', 'Compatibilidad_táctica', 'Consenso'],
        index=0
    )
    df_top15_ordenado = df_top15.sort_values(by=columna_orden, ascending=False, na_position='last')
    st.dataframe(
        df_top15_ordenado[['Ranking', 'Jugador', 'club_actual', 'Similitud', 'valor_mercado',
                            'rendimiento_esperado', 'xValue', 'Compatibilidad_táctica', 'Consenso']],
        hide_index=True,
        width='stretch'
    )

    st.divider()
    st.subheader("Generar informe PDF")
    candidato_informe = st.selectbox(
        "Elige el candidato del top-15 para el informe:",
        options=df_top15_ordenado['Jugador'].tolist(),
        key="candidato_informe_top15"
    )
    if st.button("Generar informe PDF", key="btn_generar_top15"):
        fila_ref_informe = df_jugadores[df_jugadores['Player'] == jugador_seleccionado].iloc[0]
        fila_cand_informe = df_jugadores[df_jugadores['Player'] == candidato_informe].iloc[0]
        fila_top15_cand = df_top15[df_top15['Jugador'] == candidato_informe].iloc[0]
        with st.spinner("Generando informe..."):
            pdf_bytes = generar_informe_pdf(
                fila_ref=fila_ref_informe,
                fila_cand=fila_cand_informe,
                similitud=fila_top15_cand['Similitud'],
                xvalue_cand=fila_top15_cand['xValue'],
                compatibilidad=fila_top15_cand['Compatibilidad_táctica'],
                consenso=fila_top15_cand['Consenso'],
                df_contexto=df_top15,
                ruta_fotos=RUTA_FOTOS,
                carpeta_temp=tempfile.gettempdir(),
            )
        st.session_state['pdf_bytes_top15'] = pdf_bytes
        st.session_state['pdf_nombre_top15'] = f"Informe_{jugador_seleccionado.replace(' ', '_')}_vs_{candidato_informe.replace(' ', '_')}.pdf"

    if 'pdf_bytes_top15' in st.session_state:
        st.download_button(
            "Descargar informe PDF",
            data=st.session_state['pdf_bytes_top15'],
            file_name=st.session_state['pdf_nombre_top15'],
            mime="application/pdf",
            key="download_top15"
        )
else:
    st.info("Selecciona un jugador arriba para ver sus similares.")
# ---------- COMPARADOR DIRECTO ENTRE DOS JUGADORES ----------
st.markdown("---")
st.header("Comparador directo entre dos jugadores")
st.caption("Calcula la similitud exacta entre dos jugadores y su posición en el ranking "
           "completo del grupo posicional (no solo el top-15 precomputado). Útil para "
           "casos de validación retrospectiva donde el fichaje real queda fuera del top-15.")
col1, col2 = st.columns(2)
with col1:
    jugador_a = st.selectbox("Jugador de referencia (saliente)",
                              sorted(df_jugadores['Player'].unique()), key="comp_a")
with col2:
    jugador_b = st.selectbox("Jugador a comparar (fichaje real)",
                              sorted(df_jugadores['Player'].unique()), key="comp_b")

if st.button("Comparar"):
    resultado, error = comparar_dos_jugadores(df_pca, df_jugadores, jugador_a, jugador_b)
    st.session_state['comp_resultado'] = resultado
    st.session_state['comp_error'] = error
    st.session_state['comp_jugador_a'] = jugador_a
    st.session_state['comp_jugador_b'] = jugador_b
    st.session_state.pop('pdf_bytes_comparador', None)  # la pareja pudo cambiar: descarta el PDF anterior

if 'comp_resultado' in st.session_state:
    resultado = st.session_state['comp_resultado']
    error = st.session_state['comp_error']
    jugador_a = st.session_state['comp_jugador_a']
    jugador_b = st.session_state['comp_jugador_b']

    if error:
        st.warning(error)
    else:
        # Tabla comparativa con los datos de ambos jugadores, mismo formato que el top-15
        columnas_info = ['Player', 'club_actual', 'valor_mercado', 'rendimiento_esperado', 'xValue']
        info_a = df_jugadores.loc[df_jugadores['Player'] == jugador_a, columnas_info].iloc[0]
        info_b = df_jugadores.loc[df_jugadores['Player'] == jugador_b, columnas_info].iloc[0]
        df_comparativa = pd.DataFrame([info_a, info_b]).rename(columns={
            'Player': 'Jugador',
            'club_actual': 'Club actual',
            'valor_mercado': 'Valor de mercado',
            'rendimiento_esperado': 'Rendimiento esperado',
        })
        st.dataframe(df_comparativa, hide_index=True, width='stretch')

        club_ref = info_a['club_actual']
        club_obj = info_b['club_actual']
        compat = calcular_compatibilidad(club_ref, club_obj)

        # Consenso: no se puede calcular min-max sobre una pareja de 2 (siempre daría
        # 0 y 1, sin significado). En su lugar, se inserta al jugador comparado como
        # candidato nº16 dentro del top-15 REAL del jugador de referencia, y se
        # recalcula el Consenso sobre ese conjunto de 16 con la misma función de
        # siempre. El resultado indica qué Consenso y qué puesto habría tenido el
        # fichaje real si se hubiera evaluado junto a los 15 candidatos que el
        # modelo sí propuso.
        consenso_valor = None
        puesto_consenso = None
        total_consenso = None
        df_extendido = None
        fila_similares_a = df_similares[df_similares['Player'] == jugador_a]
        if not fila_similares_a.empty:
            fila_similares_a = fila_similares_a.iloc[0]
            filas_extendido = []
            for rank in range(1, 16):
                filas_extendido.append({
                    'Jugador': fila_similares_a[f'similar_{rank}'],
                    'Similitud': fila_similares_a[f'score_{rank}']
                })
            filas_extendido.append({'Jugador': jugador_b, 'Similitud': resultado['similitud']})
            df_extendido = pd.DataFrame(filas_extendido)
            df_extendido = df_extendido.merge(
                df_jugadores[['Player', 'club_actual', 'xValue']],
                left_on='Jugador', right_on='Player', how='left'
            ).drop(columns=['Player'])
            df_extendido = df_extendido[df_extendido['club_actual'] != club_ref]
            df_extendido['Compatibilidad_táctica'] = df_extendido['club_actual'].apply(
                lambda club_candidato: calcular_compatibilidad(club_ref, club_candidato)
            )
            df_extendido = calcular_consenso(df_extendido)
            fila_candidato = df_extendido[df_extendido['Jugador'] == jugador_b]
            if not fila_candidato.empty:
                consenso_valor = fila_candidato.iloc[0]['Consenso']
                total_consenso = len(df_extendido)
                puesto_consenso = int(
                    df_extendido['Consenso'].rank(ascending=False, method='min')
                    .loc[fila_candidato.index[0]]
                )

        # Fila con los datos que relacionan a ambos jugadores entre sí
        df_relacion = pd.DataFrame([{
            'Similitud': resultado['similitud'],
            'Posición en ranking': f"{resultado['ranking']} de {resultado['total_candidatos']}",
            'Compatibilidad_táctica': round(compat, 4) if compat is not None else None,
            'Consenso': round(consenso_valor, 4) if consenso_valor is not None else None,
            'Puesto en Consenso (junto al top-15 real)':
                f"{puesto_consenso} de {total_consenso}" if puesto_consenso is not None else "No disponible",
        }])
        st.dataframe(df_relacion, hide_index=True, width='stretch')

        st.divider()
        if consenso_valor is not None:
            if st.button("Generar informe PDF", key="btn_generar_comparador"):
                fila_ref_informe = df_jugadores[df_jugadores['Player'] == jugador_a].iloc[0]
                fila_cand_informe = df_jugadores[df_jugadores['Player'] == jugador_b].iloc[0]
                with st.spinner("Generando informe..."):
                    pdf_bytes = generar_informe_pdf(
                        fila_ref=fila_ref_informe,
                        fila_cand=fila_cand_informe,
                        similitud=resultado['similitud'],
                        xvalue_cand=info_b['xValue'],
                        compatibilidad=compat,
                        consenso=consenso_valor,
                        df_contexto=df_extendido,
                        ruta_fotos=RUTA_FOTOS,
                        carpeta_temp=tempfile.gettempdir(),
                    )
                st.session_state['pdf_bytes_comparador'] = pdf_bytes
                st.session_state['pdf_nombre_comparador'] = f"Informe_{jugador_a.replace(' ', '_')}_vs_{jugador_b.replace(' ', '_')}.pdf"

            if 'pdf_bytes_comparador' in st.session_state:
                st.download_button(
                    "Descargar informe PDF",
                    data=st.session_state['pdf_bytes_comparador'],
                    file_name=st.session_state['pdf_nombre_comparador'],
                    mime="application/pdf",
                    key="download_comparador"
                )
        else:
            st.caption("Informe PDF no disponible para esta pareja: no se pudo calcular el Consenso "
                       "(el jugador de referencia no tiene top-15 de similitud precomputado).")
