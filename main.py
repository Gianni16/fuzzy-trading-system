import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta
import skfuzzy as fuzz
from fuzzy_expert.rule import FuzzyRule
from fuzzy_expert.variable import FuzzyVariable
from fuzzy_expert.inference import DecompositionalInference
from pygad import GA
import time

################################### Functions ###############################################

# def read_series(name:str):
#     folder = os.getcwd()
#     return pd.read_csv(os.path.join(folder, f'{name}_weekly.csv'))

def read_series(name:str):
    folder = os.path.join(os.getcwd(), 'data')
    filepath = os.path.join(folder, f'{name}_weekly.csv')
    return pd.read_csv(filepath)

def technical_append(time_series,indicator:list[str]):
    time_series = time_series.copy()
    if 'STO' in indicator:
        STO_k = 14
        STO_d = 3
        time_series = pd.concat([time_series,time_series.ta.stoch(high='High', low='Low', k=STO_k, d=STO_d)],axis = 1)
        del time_series[f'STOCHd_{STO_k}_{STO_d}_{STO_d}']
        time_series = time_series.rename(columns={f'STOCHk_{STO_k}_{STO_d}_{STO_d}':'STO'})    
    time_series = time_series[['Date','Close','STO']]
    if 'RSI' in indicator:
        rsi_length = 14
        time_series = pd.concat([time_series,time_series.ta.rsi(close='Close', length=rsi_length)],axis = 1)
        time_series = time_series.rename(columns={f'RSI_{rsi_length}':'RSI'})
    if 'MACD' in indicator:
        MACD_fast = 12
        MACD_slow = 26
        MACD_signal = 9
        time_series = pd.concat([time_series,time_series.ta.macd(close='Close', fast=MACD_fast, slow=MACD_slow, signal=MACD_signal)],axis = 1)
        del time_series[f'MACDs_{MACD_fast}_{MACD_slow}_{MACD_signal}']
        del time_series[f'MACD_{MACD_fast}_{MACD_slow}_{MACD_signal}']
        time_series = time_series.rename(columns={f'MACDh_{MACD_fast}_{MACD_slow}_{MACD_signal}':'MACD'})

    return time_series.dropna().reset_index(drop=True)
        
def train_test_split(indicator_matrix: pd.DataFrame,train_split,index):
    if index == 'train':
        train = indicator_matrix[:int(len(indicator_matrix)*train_split)]
        return train
    elif index == 'test':
        test = indicator_matrix[int(len(indicator_matrix)*train_split):]
        return test


def fuzzy_variables(list_param:list):
    parameters = list_param
    variables = {
    "RSI": FuzzyVariable(
        universe_range=(RSI_range[0], RSI_range[1]),
        terms={
            "Low": [(i,fuzz.gaussmf(i,parameters[0],parameters[1])) for i in np.arange(RSI_range[0],RSI_range[1])],
            "Med": [(i,fuzz.gaussmf(i,parameters[2],parameters[3])) for i in np.arange(RSI_range[0],RSI_range[1])],
            "High": [(i,fuzz.gaussmf(i,parameters[4],parameters[5])) for i in np.arange(RSI_range[0],RSI_range[1])],
        },
    ),
    "MACD": FuzzyVariable(
        universe_range=(MACD_range[0], MACD_range[1]),
        terms={
            "Low": [(i,fuzz.gaussmf(i,parameters[6],parameters[7])) for i in np.arange(MACD_range[0],MACD_range[1])],
            "Med": [(i,fuzz.gaussmf(i,parameters[8],parameters[9])) for i in np.arange(MACD_range[0],MACD_range[1])],
            "High": [(i,fuzz.gaussmf(i,parameters[10],parameters[11])) for i in np.arange(MACD_range[0],MACD_range[1])],
        },
    ),
    "STO": FuzzyVariable(
        universe_range=(STO_range[0], STO_range[1]),
        terms={
            "Low": [(i,fuzz.gaussmf(i,parameters[12],parameters[13])) for i in np.arange(STO_range[0],STO_range[1])],
            "Med": [(i,fuzz.gaussmf(i,parameters[14],parameters[15])) for i in np.arange(STO_range[0],STO_range[1])],
            "High": [(i,fuzz.gaussmf(i,parameters[16],parameters[17])) for i in np.arange(STO_range[0],STO_range[1])],
        },
    ),
    "Decision": FuzzyVariable(
        universe_range=(Decision_range[0], Decision_range[1]),
        terms={
            "Sell": [(element,fuzz.trapmf(np.arange(Decision_range[0], Decision_range[1]), Decision_Sell)[i]) for i,element in enumerate(np.arange(Decision_range[0], Decision_range[1]))],
            "Buy": [(element,fuzz.trapmf(np.arange(Decision_range[0], Decision_range[1]), Decision_Buy)[i]) for i,element in enumerate(np.arange(Decision_range[0], Decision_range[1]))],
 
         },
    )}

    return variables

def Defuzzification(frame: pd.DataFrame ,fuzzy_parameters:list):
    frame['Defuzzification'] = frame.apply(lambda x: model(
    variables=fuzzy_variables(fuzzy_parameters),
    rules=rules,
    MACD = x['MACD'], 
    RSI=x['RSI'],
    STO=x['STO'])[0]['Decision'], axis=1)
    return frame


def total_gain(data_frame: pd.DataFrame, treshold: float):
    data_frame = data_frame.copy()
    data_frame["Direction"] = 0
    data_frame["Direction"] = np.select(
        condlist=[
            data_frame["Defuzzification"] < treshold,
            data_frame["Defuzzification"] >= treshold
        ], choicelist=[-1, 1], default=0
    )
    data_frame['Direction'] = data_frame['Direction'].replace(to_replace=0, method='ffill')
    data_frame['Enter_price'] = data_frame.loc[data_frame['Direction']!=data_frame['Direction'].shift(1), 'Close']
    data_frame['Enter_price'] = data_frame['Enter_price'].fillna(method="ffill")
    data_frame.dropna(inplace=True)
    data_frame['Long'] = np.select(
        condlist=[
            (data_frame['Direction'] > 0) & (data_frame['Direction'].shift(1) != 0) & \
                            (data_frame['Direction'].shift(1) != np.nan) & \
                          (data_frame['Direction'].shift(-1) < 0)
        ], choicelist=[
            ((data_frame["Enter_price"].shift(-1) - data_frame["Enter_price"])/data_frame["Enter_price"])*100,
        ]
    )
    data_frame['Short'] = np.select(
        condlist=[
            (data_frame['Direction'] < 0) & (data_frame['Direction'].shift(1) != 0) & \
                            (data_frame['Direction'].shift(1) != np.nan) & \
                          (data_frame['Direction'].shift(-1) > 0)
        ], choicelist=[
            ((data_frame["Enter_price"] - data_frame["Enter_price"].shift(-1))/data_frame["Enter_price"])*100
        ]
    )
    data_frame['Equity_long'] = data_frame['Long'].cumsum()
    data_frame['Equity_short'] = data_frame['Short'].cumsum()
    data_frame['Gain'] = data_frame['Equity_long']+data_frame['Equity_short']
    return data_frame

############################################## Fuzzy Rules ######################################

rules = [
    FuzzyRule(
        premise=[
            ("MACD", "High"),
            ("AND","RSI", "Low"),
            ("AND","STO", "Low"),

        ],
        consequence=[("Decision", "Buy")],
    ),
    FuzzyRule(
        premise=[
            ("MACD", "Low"),
            ("AND","RSI", "High"),
            ("AND","STO", "High"),

        ],
        consequence=[("Decision", "Buy")],
    ),
    FuzzyRule(
        premise=[
            ("MACD", "High"),
            ("AND","RSI", "Med"),
            ("AND","STO", "Med"),
            
        ],
        consequence=[("Decision", "Buy")],
    ),
    FuzzyRule(
        premise=[
            ("MACD", "Low"),
            ("AND","RSI", "Med"),
            ("AND","STO", "High"),
        ],
        consequence=[("Decision", "Sell")],
    ),
    FuzzyRule(
        premise=[
            ("RSI", "High"),
            ("AND","STO", "Low"),
        ],
        consequence=[("Decision", "Sell")],
    ),
    FuzzyRule(
        premise=[
            ("MACD", "Low"),
            ("AND","RSI", "High"),
            ("AND","STO", "High"),
        ],
        consequence=[("Decision", "Sell")])]

model = DecompositionalInference(
    and_operator="min",
    or_operator="max",
    implication_operator = 'Rc',
    composition_operator="max-min",
    production_link="max",
    defuzzification_operator="cog",
)

############################################# Variable Definition ##########################################

list_series = ['Nasdaq','Djow','Ftsemib','Sp500','Vix']
index = 0
list_indicators = ['RSI','MACD','STO']
mode = 'train'
split_percentage = 0.8
technical_indicators = technical_append(read_series(list_series[index]),list_indicators)
train = train_test_split(technical_indicators,split_percentage,mode)

variation_coeff = 0.2

#Decision
Decision_range = [-10, 10]

Decision_Sell = [-10,-10,-3,3]
Decision_Buy = [-3,3,10,10]


#RSI
RSI_range = [0,100]

RSI_Low = [RSI_range[0],variation_coeff*(RSI_range[1]-RSI_range[0])]
RSI_Med = [0.5*(RSI_range[0]+RSI_range[1]),variation_coeff*(RSI_range[1]-RSI_range[0])]
RSI_High = [RSI_range[1],variation_coeff*(RSI_range[1]-RSI_range[0])]

#MACD
MACD_range = [int(train['MACD'].min()), int(train['MACD'].max())]

MACD_Low = [MACD_range[0],variation_coeff*(MACD_range[1]-MACD_range[0])]
MACD_Med = [0.5*(MACD_range[0]+MACD_range[1]),variation_coeff*(MACD_range[1]-MACD_range[0])]
MACD_High = [MACD_range[1],variation_coeff*(MACD_range[1]-MACD_range[0])]

#STO
STO_range = [0,100]

STO_Low = [STO_range[0],variation_coeff*(STO_range[1]-STO_range[0])]
STO_Med = [0.5*(STO_range[0]+STO_range[1]),variation_coeff*(STO_range[1]-STO_range[0])]
STO_High = [STO_range[1],variation_coeff*(STO_range[1]-STO_range[0])]

#treshold
treshold = 0

#parameters
num_generations = 2
sol_per_pop = 1

initial_solution = [[RSI_Low[0],RSI_Low[1],RSI_Med[0],RSI_Med[1],RSI_High[0],RSI_High[1],
                    MACD_Low[0],MACD_Low [1],MACD_Med[0],MACD_Med[1],MACD_High[0],MACD_High[1],
                    STO_Low[0],STO_Low[1],STO_Med[0],STO_Med[1],STO_High[0],STO_High[1],
                    treshold], [RSI_Low[0],RSI_Low[1],RSI_Med[0],RSI_Med[1],RSI_High[0],RSI_High[1],
                    MACD_Low[0],MACD_Low [1],MACD_Med[0],MACD_Med[1],MACD_High[0],MACD_High[1],
                    STO_Low[0],STO_Low[1],STO_Med[0],STO_Med[1],STO_High[0],STO_High[1],
                    treshold]]
num_genes = len(initial_solution[0])
num_parents_mating = 2

############################################# Model Optimization ####################################

def fitness_func(ga_instance, list_param, solution_idx):

    train_frame = Defuzzification(train,list_param)
    
    return total_gain(train_frame, list_param[-1])['Gain'][len(train_frame)-1]


start_time = time.time()
print('Start train')

ga = GA(
        num_generations=num_generations,
        initial_population = initial_solution,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_func)

ga.run()
end_time = time.time()
print(f'End_train in {end_time-start_time} seconds')
solution, solution_fitness, solution_idx = ga.best_solution()

mode = 'test'
test_frame = train_test_split(technical_indicators,split_percentage,mode)
test = Defuzzification(test_frame,solution)
Final_frame = total_gain(pd.concat([train,test]),solution[-1]) 

############################################# Model Output ####################################

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)
ax.set_title(f'LUISS Algorithm_Fuzzy_{list_series[index]}',fontweight ='bold',fontsize=20)
ax.set_ylabel('Gain', fontweight = 'bold', fontsize=20)
ax.set_xlabel('Time', fontweight = 'bold', fontsize=20)
time_col = pd.to_datetime(Final_frame['Date'])
norm_factor = Final_frame['Gain'][len(Final_frame)-1]/Final_frame['Close'].max()
plt.plot(time_col, Final_frame['Gain'], alpha=0.2, color='black',linewidth=5, label = 'General Equity')
plt.plot(time_col, Final_frame['Close']*norm_factor, alpha=0.2, color='orange',linewidth=5, label = "Close")
plt.plot(time_col, Final_frame['Gain'].cummax(), alpha=0.2, color='red',linewidth=3, label = 'Maximum Equity')
plt.plot(time_col, Final_frame['Equity_long'], alpha=0.2, color='green',linewidth=3, label = 'Long Equity')
plt.plot(time_col, Final_frame['Equity_short'], alpha=0.2, color='red',linewidth=3, label = 'Short Equity')
plt.axvline(x=pd.to_datetime(train['Date'])[len(train)-1], color='red', linestyle='--', label='Train')
ax.legend()

# plt.savefig(f'{list_series[index]}.png', dpi=150)
# plt.close(fig)
# Final_frame.to_csv(f'{list_series[index]}.csv')

output_folder = os.path.join(os.getcwd(), 'outputs')
os.makedirs(output_folder, exist_ok=True)
plt.savefig(os.path.join(output_folder, f'{list_series[index]}.png'), dpi=150)
plt.close(fig)
Final_frame.to_csv(os.path.join(output_folder, f'{list_series[index]}.csv'))