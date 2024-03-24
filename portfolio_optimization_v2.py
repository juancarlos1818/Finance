import pandas as pd
import xlwings as xw
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator


# Infromación del excel sobre el cual se jala la info
EXCEL_NAME = "Inputs.xlsx"
RETORNOS_ESPERADOS = "E(R)"
PRECIOS_HISTORICOS = "Precios"
OUTPUT_FRONTERA = "Frontera"
OUTPUT_PESOS = "Pesos"

# Importar información
workbook = xw.Book(EXCEL_NAME)
retornos_esperados = workbook.sheets(RETORNOS_ESPERADOS).range("A1").options(convert=pd.DataFrame, expand="table").value
retornos_esperados = retornos_esperados.iloc[:,0:].values
precios_historicos = workbook.sheets(PRECIOS_HISTORICOS).range("A1").options(convert=pd.DataFrame, expand="table").value

NUMERO_ACTIVOS = int(len(precios_historicos.columns))

# Numero de simulaciones
n_sim = int(input("Inputar el numero de simulaciones: "))
# Ajuste de volatilidad
AJUSTE = int(input("Inputar el ajuste de volatilidad: "))
# Puntos de la frontera
PUNTOS_FRONTERA = int(input("Inputar el numero de puntos en la frontera: "))

# Sacar retornos historicos y matriz de covarianzas

retornos_historicos = precios_historicos.sort_index().pct_change()
mat_cov = retornos_historicos.cov()*AJUSTE

# Informacion de simulaciones
port_weights = np.zeros(shape=(n_sim,NUMERO_ACTIVOS))
port_volatility = np.zeros(n_sim)
port_sr = np.zeros(n_sim)
port_return = np.zeros(n_sim)

np.random.seed(1)

# Simulacion
for i in range(n_sim):
    # Pesos
    pesos = np.random.random(NUMERO_ACTIVOS)
    pesos /= np.sum(pesos)
    port_weights[i,:] = pesos
    # Retorno esperado
    exp_ret = retornos_esperados.T.dot(pesos)
    port_return[i] = exp_ret
    # Volatilidad
    exp_vol = np.sqrt(pesos.T.dot(mat_cov.dot(pesos)))
    port_volatility[i] = exp_vol
    # Sharpe Ratio
    sr = exp_ret/exp_vol
    port_sr[i] = sr

max_sr = port_sr.max()
ind = np.argmax(port_sr)
max_sr_ret = port_return[ind]
max_sr_vol = port_volatility[ind]



# ++++++++++++++++++++++++++++++++++++++++++++++++++++++ OPTIMIZACION DE PORTAFOLIO ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Obtener retorno volatilidad y sharpe ratio
def get_ret_vol_sr (pesos):
    pesos = np.array(pesos)
    ret = retornos_esperados.T.dot(pesos)[0]
    vol = np.sqrt(pesos.T.dot(mat_cov.dot(pesos)))
    sr = ret/vol
    return [ret, vol, sr]

# Como solo se permite minimizar el ratio sharpe se usa el negativo del sharpe para sacar el mayor
def neg_sr(pesos):
    return get_ret_vol_sr(pesos)[-1]*-1

# Funcion para revisar de que la suma de los pesos es 1
def check_sum(pesos):
    return np.sum(pesos) - 1

# definición de restricciones
cons = {"type" : "eq", "fun" : check_sum}
bounds = []
for i in range(NUMERO_ACTIVOS):
    bounds.append((0,1))

# Simulación para obtener los pesos optimos
peso_inicial = 1/NUMERO_ACTIVOS  # supuesto de pesos iniciales
init_guess = [peso_inicial for _ in range(NUMERO_ACTIVOS)]
opt_results = optimize.minimize(neg_sr, init_guess, constraints=cons, bounds=bounds, method='SLSQP')
optimal_weights = opt_results.x
optimal_ret = get_ret_vol_sr(optimal_weights)[0]
optimal_vol = get_ret_vol_sr(optimal_weights)[1]
optimal_sr = get_ret_vol_sr(optimal_weights)[2]

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ARMADO DE FRONTERA++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
frontier_vol = np.linspace(0, port_volatility.max(), PUNTOS_FRONTERA)
frontier_return = []
frontier_sr = []

def maximize_ret(pesos):
    return get_ret_vol_sr(pesos)[0]*-1

bounds2 = []
for i in range(NUMERO_ACTIVOS):
    bounds2.append((-1,1))

for possible_vol in frontier_vol:
    cons = ({'type':'eq','fun':check_sum},
            {'type':'eq','fun':lambda w:get_ret_vol_sr(w)[1] - possible_vol})
    result = optimize.minimize(maximize_ret, init_guess, method='SLSQP', constraints=cons, bounds=bounds2)
    frontier_return.append(get_ret_vol_sr(result.x)[0])
    frontier_sr.append(get_ret_vol_sr(result.x)[2])

# +++++++++++++++++++++++++++++++++++++++++++++++++++EXPORTAR DATA ++++++++++++++++++++++++++++++++++++++++++++++++++
headers = ["Volatilidad", "Retornos", "Sharpe"]

data_frontera = pd.DataFrame(data={"Volatilidad": frontier_vol, "Retorno": frontier_return})
data_optimo = pd.DataFrame(data={"activos": retornos_historicos.keys(), "Pesos": optimal_weights})

workbook.sheets(OUTPUT_FRONTERA).range("A1").value = data_frontera
workbook.sheets(OUTPUT_PESOS).range("A1").value = data_optimo

# +++++++++++++++++++++++++++++++++++++++++++++++++++GRAFICAR DATA ++++++++++++++++++++++++++++++++++++++++++++++++++

plt.figure(figsize=(12,6))
plt.scatter(frontier_vol,frontier_return,c=frontier_sr, cmap='plasma')
plt.scatter(port_volatility,port_return,c=port_sr, cmap='plasma')
plt.colorbar(label='Sharpe Ratio')
plt.scatter(optimal_vol,optimal_ret,c=optimal_sr, cmap='grey')
plt.xlabel('Volatility', fontsize=15)
plt.ylabel('Return', fontsize=15)
plt.title('Efficient Frontier', fontsize=15)
plt.show()

