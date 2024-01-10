from fastapi import FastAPI
import pandas as pd
import json
import numpy as np
from fastapi.responses import JSONResponse



# Se instancia la aplicación
app = FastAPI()




#Carga los datos parquet en un dataframe de pandas
Tabla_API = pd.read_parquet('Tabla_API.parquet')
funcion_UR= pd.read_parquet('funcion_UsersRecommend.parquet')
funcion_UWD= pd.read_parquet('funcion_UsersWorstDeveloper.parquet')
funcion_SA= pd.read_parquet('funcion_SentimentAnalysis.parquet')



#Funcion 1


@app.get(path='/PlayTimeGenre',           
         description = """ <font color="darkblue">
                         DESCRIPCION <br>
                         Al ingresar un genero devuelve el año con mas horas jugadas para este</p>
                        INSTRUCCIONES<br>
                        1. Haga clik en "Try it out".<br>
                        2. Ingrese el genero en la caja de texto inferior. Ejemplo de generos: Action,Adventure, Casual, Indie, Simulation,entre otros<br>
                        3. Bajar a "Resposes"  para devolver año con más horas jugadas para dicho género. 
                        </font>
                        """,
         tags=["Features"])

def PlayTimeGenre(genero: str):
    # Filtrar el DataFrame por el género específico
  genero_data = Tabla_API[Tabla_API['genres'].str.contains(genero, case=False, na=False)]

  # Verificar si se encontraron datos para el género especificado
  if genero_data.empty:
      print(f"No hay datos para el género {genero}.")
      return None

  # Encontrar el año con más horas jugadas
  max_hours_row = genero_data.groupby('year_release')['playtime_forever'].sum().idxmax()

  # Imprimir el resultado
  result = f"El año con más horas jugadas para el género {genero}: {max_hours_row}"
  return result




#Funcion 2

@app.get('/UserForGenre', 
         description = """ <font color="darkblue">
                         DESCRIPCION <br>
                         Al ingresar un genero devuelve el usuario que acumula mas horas jugadas para el genero dado y un alista de la acumulacion de horas jugadas por año. </p>
                        INSTRUCCIONES<br>
                        1. Haga clik en "Try it out".<br>
                        2. Ingrese el genero en la caja de texto inferior. Ejemplo de generos: Action,Adventure, Casual, Indie, Simulation,entre otros<br>
                        3. Bajar a "Resposes"  para ver el usuario con más horas jugadas para el genero dado y descripción por año.
                        </font>
                        """,
          tags=["Features"])

def UserForGenre(genre):
   # Filtrar el DataFrame por el género específico
   genre_data = Tabla_API[Tabla_API['genres'].str.contains(genre, case=False, na=False)]

   # Verificar si se encontraron datos para el género especificado
   if genre_data.empty:
       print(f"No hay datos para el género {genre}.")
       return None

   # Agrupar por usuario y sumar las horas jugadas
   user_playtimes = genre_data.groupby('user_id')['playtime_forever'].sum()

   # Encontrar el usuario con más horas jugadas
   user_max_playtime = user_playtimes.idxmax()

   # Obtener la acumulación de horas jugadas por año
   playtime_by_year = genre_data.groupby('year_release')['playtime_forever'].sum().reset_index()

   # Crear un diccionario con los resultados
   result = {
       'Usuario_con_mas_horas_jugadas': user_max_playtime,
       'Horas_jugadas_por_año': playtime_by_year.to_dict('records')
   }
   return result
   
   
#Funcion 3

@app.get('/UsersRecommend', 
         description = """ <font color="darkblue">
                        DESCRIPCION <br>
                         Al ingresar un año devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado. </p>
                        INSTRUCCIONES<br>
                        1. Haga clik en "Try it out".<br>
                        2. Ingrese el año en la caja de texto inferior. Ejemplo de generos: Action,Adventure, Casual, Indie, Simulation,entre otros<br>
                        3. Bajar a "Resposes"  para ver el top 3 de juegos MÁS recomendados.
                        </font>
                        """,
          tags=["Features"])

def UsersRecommend(year: int) :
 # Filtrar el DataFrame por el año específico y las recomendaciones positivas o neutrales
 year_data = funcion_UR[(funcion_UR['year_review'] == year)]
 year_data1 = funcion_UR[(funcion_UR['recommend'] == True) & (funcion_UR['sentiment_analysis'].isin([2, 1]))]
 year = year_data1.groupby('item_name')['recommend'].sum().sort_values(ascending=False).head(3)
 result = pd.DataFrame(year).to_dict()
 # Convertir el resultado a una lista de diccionarios
 result = [{"Puesto " + str(i+1) : v} for i, v in enumerate(result['recommend'])]

 return result



#Funcion 4


@app.get('/UsersWorstDeveloper/', 
         description = """ <font color="darkblue">
                        DESCRIPCION <br>
                         Al ingresar un año devuelve el top 3 de desarrolladoras con juegos MENOS recomendados por usuarios para el año dado. </p>
                        INSTRUCCIONES<br>
                        1. Haga clik en "Try it out".<br>
                        2. Ingrese el año en la caja de texto inferior. Ejemplo de generos: Action,Adventure, Casual, Indie, Simulation,entre otros<br>
                        3. Bajar a "Resposes"  para ver el top 3 de desarrolladoras MÁS recomendadas.
                        </font>
                        """,
          tags=["Features"])


def UsersWorstDeveloper(year: int):
 # Filtrar el DataFrame por el año específico y las recomendaciones positivas o neutrales
 year_data = funcion_UWD[(funcion_UWD['year_review'] == year)]
 year_data1 = funcion_UWD[(funcion_UWD['recommend'] == False) & (funcion_UWD['sentiment_analysis']== 0)]
 year= year_data1.groupby('developer')['recommend'].sum().sort_values(ascending=False).head(3)
 result = pd.DataFrame(year).to_dict()
 # Convertir el resultado a una lista de diccionarios
 result = [{"Puesto " + str(i+1) : v} for i, v in enumerate(result['recommend'])]

 return result




#Funcion 5

@app.get('/sentiment_analysis/', 
         description = """ <font color="darkblue">
                        DESCRIPCION <br>
                         Al ingresar una desarrolladora devuelve un diccionario con el nombre de la desarrolladora como llave y una lista 
                         con la cantidad total de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento como valor. </p>
                        INSTRUCCIONES<br>
                        1. Haga clik en "Try it out".<br>
                        2. Ingrese la desarrolladora  en la caja de texto inferior. Ejemplo de generos: Action,Adventure, Casual, Indie, Simulation,entre otros<br>
                        3. Bajar a "Resposes"  para el analisis de sentimientos.

                        </font>
                        """,
          tags=["Features"])

def sentiment_analysis(empresa_desarrolladora: str) -> dict:
  # Crear un diccionario que mapee los valores '0', '1' y '2' a 'Negativo', 'Neutral' y 'Positivo'
  sentiment_mapping = {0: 'Negativo', 1: 'Neutral', 2: 'Positivo'}
  # Reemplazar los valores en la columna 'sentiment_analysis'
  funcion_SA['sentiment_analysis'] = funcion_SA['sentiment_analysis'].replace(sentiment_mapping)
  # Filtrar el DataFrame por la empresa desarrolladora
  df_filtered = funcion_SA[funcion_SA['developer'] == empresa_desarrolladora]
  # Verificar si se encontraron datos para la empresa desarrolladora
  if df_filtered.empty:
      print(f"No hay datos para la empresa {empresa_desarrolladora}.")
      return None
  # Agrupar por análisis de sentimiento y contar las filas
  df_grouped = df_filtered.groupby(['developer', 'sentiment_analysis']).size().reset_index(name='Numero de registros')
  # Convertir el DataFrame agrupado en un diccionario con el nombre de la empresa desarrolladora como llave y una lista con la cantidad total de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento como valor
  sentiment_analysis = {empresa_desarrolladora: df_grouped.set_index('sentiment_analysis')['Numero de registros'].to_dict()}

  return sentiment_analysis







