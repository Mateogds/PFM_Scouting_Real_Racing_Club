"""
Módulo de generación de informes PDF de scouting (comparativa referencia vs. candidato).

Fuente única de la lógica: se usa tanto desde el notebook de prototipado
(Código/10_Informe_PDF) como desde app.py (Código/07_Streamlit), evitando
mantener dos copias que puedan divergir.

Dependencias: fpdf2 (¡no "fpdf" a secas, es un fork distinto!), matplotlib, pandas, numpy.
    pip install fpdf2
"""

import os
import re
import io

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # backend sin ventana, necesario para generar imágenes en servidor/Streamlit
import matplotlib.pyplot as plt
import matplotlib as mpl

from fpdf import FPDF
from PIL import Image


# ---------- RUTAS Y CONSTANTES ----------

# Fuente Unicode: se reutiliza la que ya trae matplotlib instalado (DejaVu Sans),
# así no hay que descargar ni gestionar un .ttf aparte. Necesaria porque las
# fuentes básicas de fpdf2 (Helvetica, Times...) son Latin-1 y no cubren todos
# los caracteres que pueden aparecer en nombres/nacionalidades (ej. apellidos
# de Europa del Este, turcos, escandinavos).
_RUTA_FUENTES_MPL = os.path.join(os.path.dirname(mpl.__file__), 'mpl-data', 'fonts', 'ttf')
RUTA_FUENTE_REGULAR = os.path.join(_RUTA_FUENTES_MPL, 'DejaVuSans.ttf')
RUTA_FUENTE_BOLD = os.path.join(_RUTA_FUENTES_MPL, 'DejaVuSans-Bold.ttf')

LIGA_NOMBRES = {
    'es La Liga': 'LaLiga',
    'es Segunda División': 'LaLiga 2',
    'eng Premier League': 'Premier League',
    'it Serie A': 'Serie A',
    'fr Ligue 1': 'Ligue 1',
    'de Bundesliga': 'Bundesliga',
}

# Reparto de las 20 columnas _pct en dos bloques futbolísticamente coherentes
# (no es un 10/10 "matemático": Crs_90 y Fld_90 son métricas de conducción/
# creación, no defensivas, así que van al bloque ofensivo aunque desequilibre
# el número de ejes). Si se prefiere el reparto simétrico, basta con mover
# esas dos claves de una lista a la otra.
METRICAS_OFENSIVAS = [
    'Gls.1_pct', 'Ast.1_pct', 'G+A.1_pct', 'G-PK.1_pct', 'G+A-PK_pct',
    'Sh/90_pct', 'SoT/90_pct', 'SoT%_pct', 'G/Sh_pct', 'G/SoT_pct',
    'Crs_90_pct', 'Off_90_pct', 'Fld_90_pct',
]
ETIQUETAS_OFENSIVAS = [
    'Goles/90', 'Asist./90', 'G+A/90', 'G-PK/90', 'G+A-PK/90',
    'Tiros/90', 'Tiros a puerta/90', '% Tiros a puerta', 'Goles/Tiro', 'Goles/TaP',
    'Centros/90', 'Fueras de juego/90', 'Faltas recibidas/90',
]

METRICAS_DEFENSIVAS = [
    'Int_90_pct', 'TklW_90_pct', 'Fls_90_pct',
    'CrdY_90_pct', 'CrdR_90_pct', '2CrdY_90_pct', 'OG_90_pct',
]
ETIQUETAS_DEFENSIVAS = [
    'Intercepciones/90', 'Entradas ganadas/90', 'Faltas cometidas/90',
    'T. amarillas/90', 'T. rojas/90', '2ª amarilla/90', 'Autogoles/90',
]

# Qué radar(es) se muestran según el grupo posicional del jugador de REFERENCIA
GRUPOS_SOLO_DEFENSIVO = {'Defensa central', 'Lateral derecho', 'Lateral izquierdo', 'Pivote'}
GRUPOS_SOLO_OFENSIVO = {'Delantero centro', 'Extremo derecho', 'Extremo izquierdo', 'Mediocentro ofensivo'}
# Mediocentro (y Portero, que no aplica aquí) -> ambos radares

# Columnas de "estadísticas de temporada" (valores reales, no percentiles) que
# se muestran en la tabla del informe — totales de temporada + minutos de
# contexto, complementando al radar (que es percentil, no valor absoluto).
COLUMNAS_TABLA_TEMPORADA = ['MP', 'Min', 'Gls_x', 'Ast', 'G+A', 'Sh', 'SoT',
                             'CrdY_y', 'CrdR_y', 'Int', 'TklW', 'Crs']
ETIQUETAS_TABLA_TEMPORADA = ['PJ', 'Min.', 'Goles', 'Asist.', 'G+A', 'Tiros', 'Tiros a puerta',
                              'T.A.', 'T.R.', 'Intercep.', 'Entradas', 'Centros']


# ---------- FOTO ----------

def obtener_ruta_foto(url_tmarkt, ruta_fotos):
    """
    Extrae el ID numérico de Transfermarkt de la URL (misma regex usada en el
    scraping: /spieler/(\\d+)) y busca la foto en disco probando extensiones
    comunes. Devuelve None si no hay URL válida o no existe el archivo
    (el informe debe poder generarse igualmente, con un hueco en vez de foto).
    """
    m = re.search(r'/spieler/(\d+)', str(url_tmarkt))
    if not m:
        return None
    id_jugador = m.group(1)
    for ext in ('.jpg', '.jpeg', '.png'):
        ruta = os.path.join(ruta_fotos, f"{id_jugador}{ext}")
        if os.path.exists(ruta):
            return ruta
    return None


# ---------- RADAR ----------

def generar_radar(jugador_ref_row, jugador_cand_row, metricas, etiquetas, titulo, ruta_salida_img):
    """
    Radar superpuesto referencia (azul) vs. candidato (rojo) sobre las
    columnas _pct indicadas. Guarda la imagen en ruta_salida_img (PNG) para
    poder insertarla en el PDF con fpdf2 (no soporta gráficos vectoriales
    matplotlib directamente, solo imágenes).
    """
    valores_ref = pd.to_numeric(jugador_ref_row[metricas], errors='coerce').fillna(0).values
    valores_cand = pd.to_numeric(jugador_cand_row[metricas], errors='coerce').fillna(0).values

    n = len(metricas)
    angulos = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angulos_cerrado = angulos + angulos[:1]
    valores_ref_cerrado = np.concatenate((valores_ref, [valores_ref[0]]))
    valores_cand_cerrado = np.concatenate((valores_cand, [valores_cand[0]]))

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.plot(angulos_cerrado, valores_ref_cerrado, linewidth=2, color='#1f4e8c',
            label=str(jugador_ref_row['Player']))
    ax.fill(angulos_cerrado, valores_ref_cerrado, color='#1f4e8c', alpha=0.15)
    ax.plot(angulos_cerrado, valores_cand_cerrado, linewidth=2, color='#c0392b',
            label=str(jugador_cand_row['Player']))
    ax.fill(angulos_cerrado, valores_cand_cerrado, color='#c0392b', alpha=0.15)

    ax.set_xticks(angulos)
    ax.set_xticklabels(etiquetas, fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25', '50', '75', '100'], fontsize=6, color='gray')
    ax.set_title(titulo, fontsize=12, weight='bold', pad=22)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(ruta_salida_img, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return ruta_salida_img


def radares_a_generar(grupo_posicional_ref):
    """Devuelve la lista de radares (clave, metricas, etiquetas, titulo) según
    el grupo posicional del jugador de referencia."""
    radares = []
    if grupo_posicional_ref not in GRUPOS_SOLO_OFENSIVO:
        radares.append(('defensivo', METRICAS_DEFENSIVAS, ETIQUETAS_DEFENSIVAS, 'Perfil defensivo (percentil)'))
    if grupo_posicional_ref not in GRUPOS_SOLO_DEFENSIVO:
        radares.append(('ofensivo', METRICAS_OFENSIVAS, ETIQUETAS_OFENSIVAS, 'Perfil ofensivo (percentil)'))
    return radares


# ---------- CONCLUSIÓN (texto de recomendación en prosa) ----------

# Métricas donde percentil MÁS ALTO es PEOR (más tarjetas, más autogoles, más
# fueras de juego, más faltas cometidas) — hay que invertir el signo de la
# diferencia antes de decidir si es "fortaleza" o "debilidad" del candidato,
# si no un jugador con más tarjetas rojas que la referencia sale descrito
# como si eso fuera un punto a favor.
METRICAS_INVERTIDAS = {
    'CrdY_90_pct', 'CrdR_90_pct', '2CrdY_90_pct', 'OG_90_pct', 'Off_90_pct', 'Fls_90_pct',
}


def _comparar_metricas_vs_referencia(fila_ref, fila_cand, grupo_ref, top_n=2, umbral=10):
    """
    Compara candidato vs. referencia en las mismas métricas _pct que se
    dibujan en el/los radar(es) de este grupo posicional (radares_a_generar),
    para que la conclusión hable exactamente de lo que el lector ve en el
    gráfico. Devuelve (fortalezas, debilidades): listas de hasta top_n
    etiquetas donde el candidato está mejor/peor que la referencia (ya
    corregido el signo en las métricas de METRICAS_INVERTIDAS).
    Si la mayor diferencia no supera `umbral` puntos de percentil, se
    considera que los perfiles están parejos y se devuelven listas vacías
    (evita forzar una comparación donde apenas hay diferencia real).
    """
    metricas, etiquetas = [], []
    for _, mets, etqs, _ in radares_a_generar(grupo_ref):
        metricas += mets
        etiquetas += etqs

    diffs = []
    for metrica, etiqueta in zip(metricas, etiquetas):
        v_ref = pd.to_numeric(fila_ref.get(metrica), errors='coerce')
        v_cand = pd.to_numeric(fila_cand.get(metrica), errors='coerce')
        if pd.isna(v_ref) or pd.isna(v_cand):
            continue
        diff = v_cand - v_ref
        if metrica in METRICAS_INVERTIDAS:
            diff = -diff
        diffs.append((etiqueta, diff))

    if not diffs or max(abs(d) for _, d in diffs) < umbral:
        return [], []

    diffs.sort(key=lambda x: x[1], reverse=True)
    fortalezas = [etq for etq, d in diffs[:top_n] if d > 0]
    debilidades = [etq for etq, d in diffs[-top_n:] if d < 0]
    return fortalezas, debilidades


def _listar_es(items):
    """'A, B y C' — para insertar listas de métricas en una frase."""
    if not items:
        return ''
    if len(items) == 1:
        return items[0]
    return ', '.join(items[:-1]) + ' y ' + items[-1]


def generar_conclusion(fila_ref, fila_cand, similitud, xvalue_cand, compatibilidad, consenso, df_contexto):
    """
    Conclusión en prosa integrada (no bloques separados por etiqueta) — el
    texto que iría bajo un título "Conclusión" en un informe de scouting
    real: similitud, xValue, compatibilidad, comparación directa frente al
    jugador de referencia (dónde es mejor / peor) y un veredicto explícito
    de fichaje, todo entretejido en 1-2 párrafos de lenguaje natural.
    """
    jugador_ref = str(fila_ref['Player'])
    jugador_cand = str(fila_cand['Player'])
    grupo_ref = fila_ref.get('grupo_posicional')

    top_consenso = df_contexto.sort_values('Consenso', ascending=False).iloc[0]
    puesto_consenso = int(df_contexto['Consenso'].rank(ascending=False, method='min')
                           .loc[df_contexto[df_contexto['Jugador'] == jugador_cand].index[0]])
    total = len(df_contexto)
    es_top_consenso = (top_consenso['Jugador'] == jugador_cand)

    fortalezas, debilidades = _comparar_metricas_vs_referencia(fila_ref, fila_cand, grupo_ref)

    # --- Similitud ---
    if similitud >= 0.85:
        frase_similitud = f"muestra una similitud estadística muy alta con {jugador_ref} ({similitud:.3f}), prácticamente un calco de perfil dentro de su grupo posicional"
    elif similitud >= 0.70:
        frase_similitud = f"comparte una similitud estadística alta con {jugador_ref} ({similitud:.3f}), coincidiendo en la mayoría de las métricas de rendimiento clave"
    elif similitud >= 0.55:
        frase_similitud = f"presenta una similitud moderada con {jugador_ref} ({similitud:.3f}) — perfil razonablemente parecido, pero no un calco"
    else:
        frase_similitud = f"tiene una similitud baja con {jugador_ref} ({similitud:.3f}), por lo que su presencia aquí se apoya más en el xValue y la compatibilidad que en el parecido estadístico puro"

    # --- Comparación directa (lo que se ve en el radar) ---
    if fortalezas or debilidades:
        frase_comparacion = f" Comparado directamente con {jugador_ref}, "
        partes = []
        if fortalezas:
            partes.append(f"destaca por encima en {_listar_es(fortalezas)}")
        if debilidades:
            partes.append(f"queda por debajo en {_listar_es(debilidades)}")
        frase_comparacion += ', aunque '.join(partes) if len(partes) == 2 else partes[0]
        frase_comparacion += '.'
    else:
        frase_comparacion = f" En las métricas que muestra el radar, ambos perfiles están muy parejos, sin diferencias destacables entre uno y otro."

    # --- xValue ---
    valor_cand = fila_cand.get('valor_mercado', 'N/D')
    if pd.isna(xvalue_cand):
        frase_xvalue = " El xValue no está disponible para este jugador."
    elif xvalue_cand >= 20:
        frase_xvalue = (f" A nivel de mercado, su xValue es muy positivo (+{xvalue_cand:.2f}): rinde claramente por "
                          f"encima de lo que sugiere su valor actual ({valor_cand}), lo que apunta a una posible "
                          f"oportunidad de mercado.")
    elif xvalue_cand > 0:
        frase_xvalue = f" Su xValue es positivo (+{xvalue_cand:.2f}), ligeramente por encima de lo que su valor de mercado ({valor_cand}) indicaría."
    elif xvalue_cand > -20:
        frase_xvalue = (f" Su xValue es negativo ({xvalue_cand:.2f}): su valor de mercado ({valor_cand}) ya iguala o "
                          f"supera su rendimiento esperado, así que no aporta margen de plusvalía por esta vía, sin "
                          f"que eso lo convierta automáticamente en mal fichaje.")
    else:
        frase_xvalue = (f" El punto débil es el xValue, muy negativo ({xvalue_cand:.2f}): su valor de mercado "
                          f"({valor_cand}) está notablemente por encima de lo que su rendimiento esperado sugiere, "
                          f"algo habitual en perfiles de prestigio pero que aquí es la principal señal de alerta.")

    # --- Compatibilidad táctica ---
    club_ref = fila_ref.get('club_actual', 'el club de origen')
    if compatibilidad is None or pd.isna(compatibilidad):
        frase_compat = f" No se ha podido calcular la compatibilidad táctica con {club_ref} (club fuera de la cobertura de datos de estilo de juego)."
    elif compatibilidad >= 0.85:
        frase_compat = f" Tácticamente encaja muy bien con el estilo de {club_ref} ({compatibilidad:.3f}), por lo que la adaptación de estilo debería ser mínima."
    elif compatibilidad >= 0.70:
        frase_compat = f" La compatibilidad táctica con {club_ref} es buena ({compatibilidad:.3f}), sin choque de estilos relevante."
    elif compatibilidad >= 0.50:
        frase_compat = f" La compatibilidad táctica con {club_ref} es solo moderada ({compatibilidad:.3f}), así que podría necesitar un periodo de adaptación al estilo de juego."
    else:
        frase_compat = f" La compatibilidad táctica con {club_ref} es baja ({compatibilidad:.3f}), lo que es un riesgo real de adaptación más allá de lo puramente estadístico."

    # --- Consenso + veredicto ---
    if es_top_consenso:
        frase_consenso = f" Dentro del conjunto de {total} candidatos evaluados, es además la recomendación de consenso número uno, combinando el mejor equilibrio entre similitud, xValue y compatibilidad."
    else:
        frase_consenso = f" Dentro del conjunto de {total} candidatos evaluados ocupa la posición {puesto_consenso} por score de consenso; {top_consenso['Jugador']} encabeza ese ranking combinado."

    if consenso >= 0.75 and (pd.isna(xvalue_cand) or xvalue_cand > -20):
        veredicto = f" En conjunto, {jugador_cand} es un fichaje recomendable para sustituir a {jugador_ref}: perfil suficientemente parecido, sin señales de alerta relevantes en valor de mercado ni en encaje táctico."
    elif consenso >= 0.55:
        veredicto = f" En conjunto, {jugador_cand} es una opción razonable pero no evidente: el perfil convence, aunque {'el xValue' if (pd.notna(xvalue_cand) and xvalue_cand <= -20) else 'algún aspecto del encaje'} obliga a valorarlo con matices antes de recomendar el fichaje sin reservas."
    else:
        veredicto = f" En conjunto, no se recomienda priorizar el fichaje de {jugador_cand} frente a otras opciones de este top-15: el balance entre similitud, valor de mercado y encaje táctico no juega a su favor."

    return (
        f"{jugador_cand} ({fila_cand.get('club_actual', '—')}) {frase_similitud}."
        f"{frase_comparacion}"
        f"{frase_xvalue}"
        f"{frase_compat}"
        f"{frase_consenso}"
        f"{veredicto}"
    )


# ---------- ENSAMBLADO DEL PDF ----------

class _InformePDF(FPDF):
    def __init__(self):
        super().__init__(orientation='P', unit='mm', format='A4')
        self.add_font('DejaVu', '', RUTA_FUENTE_REGULAR)
        self.add_font('DejaVu', 'B', RUTA_FUENTE_BOLD)
        self.set_auto_page_break(auto=True, margin=15)

    def footer(self):
        self.set_y(-12)
        self.set_font('DejaVu', '', 7)
        self.set_text_color(140, 140, 140)
        self.cell(0, 8, f"Informe generado automáticamente — PFM Scouting por Similitud (Real Racing Club de Santander) — página {self.page_no()}",
                  align='C')


def _valor_o_guion(fila, col, sufijo=''):
    v = fila.get(col, None)
    if pd.isna(v):
        return '—'
    return f"{v}{sufijo}"


def _limpiar_texto(v):
    """
    Normaliza texto libre de Transfermarkt para mostrarlo en el informe.
    Hallazgo real en el dataset: 1076/2666 jugadores (dobles nacionalidades)
    tienen 'nacionalidad' separada por espacios de no-separación (\\xa0\\xa0)
    en vez de una coma — ej. 'Inglaterra\\xa0\\xa0Ghana' — que en PDF se ve
    como un hueco extraño en vez de "Inglaterra, Ghana". Se corrige aquí en
    la capa de presentación; si en algún momento se reprocesa el CSV fuente,
    esto puede limpiarse también en el pipeline de datos.
    """
    if pd.isna(v):
        return '—'
    texto = re.sub(r'\s*\xa0+\s*', ', ', str(v))
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto


def _fmt_edad(fila):
    """
    Hallazgo real en el dataset: 30 de 2666 jugadores (todos del bloque de
    Real Racing Club añadido aparte del merge Big5) tienen 'Age' sin limpiar
    en el formato crudo de FBref 'años-días' (ej. '21-139' = 21 años y 139
    días), mientras el resto del dataset ya quedó como año simple ('26').
    Se muestra siempre solo el año, tomando la parte antes del guion si
    existe.
    """
    v = fila.get('Age', None)
    if pd.isna(v):
        return '—'
    return str(v).split('-')[0].strip()


def _dibujar_ficha_jugador(pdf, fila, ruta_fotos, x, ancho):
    """Dibuja la tarjeta de un jugador (foto + datos básicos) en la posición x indicada."""
    y_inicio = pdf.get_y()
    ruta_foto = obtener_ruta_foto(fila.get('UrlTmarkt'), ruta_fotos)
    caja_ancho = ancho * 0.4
    caja_alto = 42
    if ruta_foto:
        # No se fuerzan w y h a la vez (fpdf2 estiraría la imagen y
        # deformaría la cara) — se calcula el tamaño real para encajarla
        # dentro de la caja preservando proporción, centrada.
        try:
            with Image.open(ruta_foto) as im:
                ancho_px, alto_px = im.size
            ratio = min(caja_ancho / ancho_px, caja_alto / alto_px)
            w_final, h_final = ancho_px * ratio, alto_px * ratio
            x_foto = x + (caja_ancho - w_final) / 2
            y_foto = y_inicio + (caja_alto - h_final) / 2
            pdf.image(ruta_foto, x=x_foto, y=y_foto, w=w_final, h=h_final)
        except Exception:
            ruta_foto = None
    if not ruta_foto:
        pdf.set_xy(x, y_inicio)
        pdf.set_draw_color(200, 200, 200)
        pdf.cell(caja_ancho, caja_alto, '', border=1)

    x_texto = x + ancho * 0.4 + 3
    ancho_texto = ancho * 0.6 - 3
    pdf.set_xy(x_texto, y_inicio)
    pdf.set_font('DejaVu', 'B', 10)
    nombre_usual = str(fila.get('Player', '—'))
    pdf.multi_cell(ancho_texto, 5, nombre_usual, align='L')

    # nombre_completo (Transfermarkt) como dato complementario, solo si aporta
    # algo distinto del nombre usual (Player, de FBref) que ya se ha mostrado
    nombre_completo = _limpiar_texto(fila.get('nombre_completo'))
    if nombre_completo != '—' and nombre_completo.strip().lower() != nombre_usual.strip().lower():
        pdf.set_x(x_texto)
        pdf.set_font('DejaVu', '', 7)
        pdf.set_text_color(120, 120, 120)
        pdf.multi_cell(ancho_texto, 3.5, nombre_completo, align='L')
        pdf.set_text_color(20, 20, 20)

    pdf.set_font('DejaVu', '', 8)
    liga = LIGA_NOMBRES.get(fila.get('Comp'), fila.get('Comp', '—'))
    lineas = [
        f"Club: {_valor_o_guion(fila, 'club_actual')}",
        f"Liga: {liga}",
        f"Edad: {_fmt_edad(fila)}",
        f"Nacionalidad: {_limpiar_texto(fila.get('nacionalidad'))}",
        f"Posición: {_valor_o_guion(fila, 'posicion')}",
        f"Valor de mercado: {_valor_o_guion(fila, 'valor_mercado')}",
    ]
    for linea in lineas:
        pdf.set_x(x_texto)
        pdf.multi_cell(ancho_texto, 4.2, linea, align='L')

    pdf.set_y(y_inicio + caja_alto + 4)


def _dibujar_tabla_temporada(pdf, fila_ref, fila_cand):
    pdf.set_font('DejaVu', 'B', 9)
    pdf.cell(0, 6, 'Estadísticas de temporada', ln=True)
    pdf.set_font('DejaVu', '', 7.5)

    ancho_etiqueta = 55
    ancho_col = (190 - ancho_etiqueta) / 2

    pdf.set_fill_color(235, 235, 235)
    pdf.cell(ancho_etiqueta, 5.5, '', border=1, fill=True)
    pdf.cell(ancho_col, 5.5, str(fila_ref['Player'])[:22], border=1, align='C', fill=True)
    pdf.cell(ancho_col, 5.5, str(fila_cand['Player'])[:22], border=1, align='C', fill=True)
    pdf.ln()

    for col, etiqueta in zip(COLUMNAS_TABLA_TEMPORADA, ETIQUETAS_TABLA_TEMPORADA):
        pdf.cell(ancho_etiqueta, 5.5, etiqueta, border=1)
        pdf.cell(ancho_col, 5.5, _valor_o_guion(fila_ref, col), border=1, align='C')
        pdf.cell(ancho_col, 5.5, _valor_o_guion(fila_cand, col), border=1, align='C')
        pdf.ln()
    pdf.ln(3)


def generar_informe_pdf(fila_ref, fila_cand, similitud, xvalue_cand, compatibilidad, consenso,
                         df_contexto, ruta_fotos, carpeta_temp, ruta_salida=None):
    """
    Genera el informe PDF comparativo referencia vs. candidato.

    fila_ref, fila_cand: filas (pandas.Series) de jugadores_features.csv
    similitud, xvalue_cand, compatibilidad, consenso: métricas ya calculadas para este par
    df_contexto: DataFrame (top-15 o top-16) con columnas Jugador/Similitud/xValue/
                 Compatibilidad_táctica/Consenso, para generar_texto_consenso
    ruta_fotos: carpeta con las fotos (Datos/06_Fotos_PDF)
    carpeta_temp: carpeta donde guardar temporalmente las imágenes de los radares
    ruta_salida: si se indica, además de devolver los bytes, guarda el PDF en esta ruta

    Devuelve: bytes del PDF (listos para st.download_button o para escribir a disco)
    """
    os.makedirs(carpeta_temp, exist_ok=True)

    pdf = _InformePDF()
    pdf.add_page()

    pdf.set_font('DejaVu', 'B', 14)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 10, 'Informe de Scouting — Comparativa de jugadores', ln=True, align='C')
    pdf.set_font('DejaVu', '', 9)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 6, f"Referencia: {fila_ref['Player']}   |   Candidato: {fila_cand['Player']}", ln=True, align='C')
    pdf.set_text_color(20, 20, 20)
    pdf.ln(4)

    # Fichas de ambos jugadores, lado a lado
    y_fichas = pdf.get_y()
    _dibujar_ficha_jugador(pdf, fila_ref, ruta_fotos, x=10, ancho=90)
    pdf.set_y(y_fichas)
    _dibujar_ficha_jugador(pdf, fila_cand, ruta_fotos, x=105, ancho=90)
    pdf.ln(2)

    # Métricas de relación (similitud, xValue, compatibilidad, consenso)
    pdf.set_font('DejaVu', 'B', 9)
    pdf.cell(0, 6, 'Métricas de comparación', ln=True)
    pdf.set_font('DejaVu', '', 8)
    compat_txt = f"{compatibilidad:.3f}" if compatibilidad is not None and pd.notna(compatibilidad) else 'N/D'
    pdf.cell(0, 5.5,
             f"Similitud coseno: {similitud:.4f}     xValue candidato: {xvalue_cand:.2f}     "
             f"Compatibilidad táctica: {compat_txt}     Score de consenso: {consenso:.4f}",
             ln=True)
    pdf.ln(2)

    _dibujar_tabla_temporada(pdf, fila_ref, fila_cand)

    # Radares (según grupo posicional de la referencia)
    lista_radares = radares_a_generar(fila_ref.get('grupo_posicional'))
    rutas_radar = []
    for clave, metricas, etiquetas, titulo in lista_radares:
        ruta_img = os.path.join(carpeta_temp, f"_radar_{clave}_tmp.png")
        generar_radar(fila_ref, fila_cand, metricas, etiquetas, titulo, ruta_img)
        rutas_radar.append(ruta_img)

    y_radar = pdf.get_y()
    if len(rutas_radar) == 1:
        pdf.image(rutas_radar[0], x=55, y=y_radar, w=100)
        pdf.set_y(y_radar + 95)
    elif len(rutas_radar) == 2:
        pdf.image(rutas_radar[0], x=8, y=y_radar, w=95)
        pdf.image(rutas_radar[1], x=107, y=y_radar, w=95)
        pdf.set_y(y_radar + 90)

    # Conclusión: párrafo único en prosa (no bloques separados por etiqueta),
    # con veredicto explícito y comparación directa frente al jugador de
    # referencia — siempre en página nueva para que quede como sección
    # propia y legible, en vez de intentar aprovechar el hueco bajo el radar.
    pdf.add_page()
    pdf.set_font('DejaVu', 'B', 13)
    pdf.cell(0, 8, 'Conclusión', ln=True)
    pdf.ln(1)

    texto_conclusion = generar_conclusion(
        fila_ref, fila_cand, similitud, xvalue_cand, compatibilidad, consenso, df_contexto
    )
    pdf.set_font('DejaVu', '', 9)
    pdf.multi_cell(0, 5.3, texto_conclusion, align='J')

    for ruta_img in rutas_radar:
        try:
            os.remove(ruta_img)
        except OSError:
            pass

    salida = bytes(pdf.output())
    if ruta_salida:
        carpeta_salida = os.path.dirname(ruta_salida)
        if carpeta_salida:
            os.makedirs(carpeta_salida, exist_ok=True)
        with open(ruta_salida, 'wb') as f:
            f.write(salida)
    return salida
