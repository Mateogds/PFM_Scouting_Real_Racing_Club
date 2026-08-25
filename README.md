# PFM — Scouting por Similitud (Real Racing Club de Santander)

Herramienta de scouting por similitud de jugadores desarrollada como Trabajo Fin de Máster (Big Data Deportivo, Sports Data Campus / UCAM). Dado un jugador de referencia, devuelve un top-15 de jugadores estadísticamente similares, cruzado con un indicador de valor esperado (xValue) y un índice de compatibilidad táctica entre clubes. Caso de estudio: Real Racing Club de Santander.

**Aplicación en producción:** https://pfmscoutingrealracingclub-no6wkakcrqgx6khcke9fgh.streamlit.app/

## Qué hace

- **Similitud estadística:** percentiles por grupo posicional → PCA → similitud coseno.
- **xValue:** rendimiento esperado del jugador frente a su valor de mercado (Transfermarkt).
- **Compatibilidad táctica:** similitud de estilo de juego entre el club de origen y el de destino (métricas DataGaffer).
- **Informes en PDF:** ficha del jugador, radar comparativo y conclusión automática de fortalezas/debilidades, generados bajo demanda.

## Cómo ejecutarlo en local

```bash
git clone https://github.com/Mateogds/PFM_Scouting_Real_Racing_Club.git
cd PFM_Scouting_Real_Racing_Club/Codigo/07_Streamlit
pip install -r requirements.txt
streamlit run app.py
```
> Si no tienes git instalado, puedes descargar el proyecto directamente desde el botón **Code → Download ZIP** de esta página, descomprimirlo, y continuar desde el segundo comando (`cd`) usando la ruta donde lo hayas descomprimido.

Dependencias principales: streamlit 1.51.0, pandas 2.3.3, numpy 2.3.5, scikit-learn 1.7.2, fpdf2 2.8.8, matplotlib 3.10.6, Pillow 12.0.0.

## Estructura del repositorio

- `Codigo/07_Streamlit` — código de la aplicación.
- `Datos` — datasets utilizados por el modelo.

## Validación

El sistema fue validado retrospectivamente contra 10 fichajes reales del verano de 2026: 3 aciertos (30%) dentro del criterio top-15, frente a una probabilidad de acierto aleatorio de aproximadamente el 5%. Detalle completo en la memoria del proyecto.

## Autor

Mateo Gandarillas Seco — Máster en Big Data Deportivo, Sports Data Campus / UCAM.
