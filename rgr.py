import numpy as np
import matplotlib.pyplot as plt
bin_name = []
bin_name_full = []
name = input("введите имя латиницей >> ")
surname = input("Введите фамилию латинницей >> ")
full_name = name+surname
#full_name_2 = "KirillKulakov"
full_name_returnd = ""
polinom = [1,0,1,1,1,0,1,1]

register_x = [0,1,0,1,1]
register_y = [1,0,0,1,0]

print("Вас зовут: ", full_name)


def ASCII_coder_to_bin(full_name): #кодирует строку в двоичную систему
    for i in full_name:
        bin_name.append(bin(ord(i))[2:]) #Переводим каждый символ в десятичное число по ASCII, затем это число переводим в двоичное значение
    for i in bin_name:
        for j in i:
            bin_name_full.append(int(j)) #Заполняем список нашими битами
        
def ASCII_coder_to_char(bin_nam): #декодирует биты в текст
    full_name_returnd = ""
    for i in range(len(bin_nam)//7): #7 - это количество бит, занимаемые одним символом в двоичном виде
        name = ""
        for j in range(i*7, (i*7)+7): #каждый раз проходимся по 7 бити накапливаем строку, состоящую из 7 бит
            name += str(bin_nam[j])   
        full_name_returnd += chr(int(str(name),2)) #Сначала переводим двоичное число в десятичное, затем десятичное число в символ по ASCII
    return full_name_returnd

def calc_CRC(polinom,exten_data, result, LEN_G, N):#Расчет CRC
    temp = exten_data.copy()
    for i in range(N):
        if temp[i] == 1: #пропускаем нули
            for j in range(LEN_G):
                temp[j+i] = temp[j+i] ^ polinom[j] #XOR
    for i in range(LEN_G-1):
        result[i] = temp[((LEN_G-1 + N)- LEN_G) + (i+1)] #заполняем массив CRC


def reg_x(register_x): #сдвиг регистра Х
    tmp = (register_x[2] + register_x[3])%2
    register_x = np.roll(register_x,1)
    register_x[0] = tmp
    return register_x
def reg_y(register_y): #сдвиг регистра Y
    tmp = (register_y[1] + register_y[2])%2
    register_y = np.roll(register_y,1)
    register_y[0] = tmp
    return register_y

def GoldSeq(register_x, register_y, seq, len_pos):#Генерация последовательности голда
    for i in range(len_pos):
        seq[i] = (register_x[4] + register_y[4])%2
        
        register_x = reg_x(register_x)
        register_y = reg_y(register_y)
        
def corr(gold, signal, LEN_S):#Корреляционный прием
    max_corr = -100.2
    max_ind = 0
    temp = np.copy(signal) #копируем сигнал, что бы он не изменился 
    temp2 = np.full(len(gold), fill_value=int(0)) #массив для хранения последовательности
    for i in range(len(temp)): #С каждой итерацией смещаем массив, и считаем корреляцию с последователностю голда
        temp = np.roll(signal, -i)
        temp2 = temp[0:len(gold)]
        cor= np.correlate(temp2, gold)
        if cor > max_corr:#Находим максимальную корреляцию и индекс сдвига
            max_corr = cor
            max_ind = i
    itog = signal[max_ind:] #отсекаем все лишнее до нашего массива
    return itog
            
def interpretator(signal, N, L, M, G):#интерпретировние данных в биты
    temp = np.full(N, fill_value=float(0)) #массив для хранения буфера N отчетов
    temp2 = np.copy(signal) 
    result = np.full(L+G+M, fill_value=int(0))# Результат
    for i in range(L+G+M): 
        for j in range(N): #набираем по N отчетов
            temp[j] = temp2[j]
        temp2 = np.roll(temp2, -N) #сдвигаем массив на N элементов
        if np.mean(temp) >= 0.5: #если среднее значение всех элементов буфера больше 0.5, то значит передавалась 1, инчае 0
           result[i] = 1
    
        else:
            result[i] = 0
    return result[G:]
            
            
    
#Кодируем строку в биты
ASCII_coder_to_bin(full_name)
#переставляем их из списка в массив
array = np.array(bin_name_full)
#print(array)

L = len(array) #Длина данных
M = len(polinom)-1 #Длина CRC
LEN_G = len(polinom)#Длина полинома
exten_N = L+M #Длина расширенного массива

#3 пункт

#объявляем массив длинной L + M
exten_data = np.full(exten_N, fill_value=int(0))
#заполняем расширенный массив данными без CRC
for i in range(len(array)):
    exten_data[i] = array[i]

#Объявляем массив для CRC
CRC = np.full(M, fill_value=int(0))
#Счтаем CRC
calc_CRC(polinom, exten_data, CRC, LEN_G, L)
print("CRC = ",CRC)
#Заполняем оставшиеся места в массиве данными CRC
for i in range(L, exten_N):
    exten_data[i] = CRC[i-L]


G = len_pos = 2**5 -1 #Длина последовательности голда
#объявляем массив для голда
seq = np.full(len_pos, fill_value=int(0))
#Генерируем последовательность голда
GoldSeq(register_x, register_y, seq, len_pos)
#print(seq)
#объеденяем два массива вместе, получая Gold+Data+CRC
exten_data_full = np.concatenate((seq, exten_data), axis=0)
#print("Gold+data+CRC",exten_data_full)


N = 8 #количество отчетов на 1 бит
samples = np.repeat(exten_data_full,N) #Увеличиваем количество каждого бита на N
LEN_S = len(samples) 



#Выводим графики
t = np.arange(0,L)
t2 = np.arange(0, exten_N + G)
t3 = np.arange(0, len(samples))
plt.figure(figsize=(13, 20))
plt.subplot(4, 1, 1)

plt.plot(t, array)
plt.xlabel('элемент массива')
plt.ylabel('значение')
plt.title("data")

plt.subplot(4, 1, 2)
plt.plot(t2, exten_data_full)
plt.xlabel('элемент массива')
plt.ylabel('значение')
plt.title("Gold+data+CRC")
    
plt.subplot(4, 1, 3)
plt.plot(t3, samples)
plt.xlabel('элемент массива')
plt.ylabel('значение')
plt.title("Samples")


#Объявляем массив для сигнала
Signal = np.full(2*N*(exten_N + G), fill_value=float(0))

print("введите число от 0 до", len(samples))
ot = int(input(">> "))

for i in range(ot, ot+len(samples)):#вставляем наш масив на место ot
    Signal[i]=samples[i-ot]

    

#Сохраняем наш массив
Signal_save = np.copy(Signal)

#сигма для нормального распределения
q=float(input("Введите q >> "))
noice = np.random.normal(0, q, len(Signal)) #Генерирует шум
#Складываем сигнал и шум
for i in range(len(Signal)):
    Signal[i] = Signal[i] + noice[i]



#Увеличенная в N бит последовательность голда 
ex_seq = np.repeat(seq,N)

#Корреляционный прием
itog_signal = corr(ex_seq, Signal, LEN_S) 

#Вывод графиков
t4 = np.arange(0, len(samples)*2)
t5 = np.arange(0, len(itog_signal))

plt.figure(2,figsize=(13, 20))    
plt.subplot(4, 1, 1)
plt.plot(t4, Signal_save)
plt.xlabel('элемент массива')
plt.ylabel('значение')
plt.title("Signal")

plt.subplot(4, 1, 2)
plt.plot(t4, noice)
plt.xlabel('элемент массива')
plt.ylabel('значение')
plt.title("Noice")    

plt.subplot(4, 1, 3)
plt.plot(t4, Signal)
plt.xlabel('элемент массива')
plt.ylabel('значение')
plt.title("Signal+Noice")    

plt.subplot(4, 1, 4)
plt.plot(t5, itog_signal)
plt.xlabel('элемент массива')
plt.ylabel('значение')
plt.title("Cut signal")

#Интерпретируем наш сигнал в биты
return_signal = interpretator(itog_signal, N, L, M, G)
#проверяем на ошибки
return_CRC = np.full(M, fill_value=int(0))
calc_CRC(polinom, return_signal, return_CRC, LEN_G, L)
if np.mean(return_CRC) > 0: #Если есть хотя бы одна единичка, значит есть ошибка
    print("Найдены ошибки");
else:
    print("Ошибки не обнаружены")
    return_data = return_signal[0:L] #обрезаем CRC
    full_name_returnd = ASCII_coder_to_char(return_data) #Декодируем сообщение
    print("\n\nПолученные данные", full_name_returnd)       

    ch=0 #Если 1, до будет применен fftshift
    
    #Находим спектр сигналов
    spect_signal = np.fft.fft(Signal_save)
    spect_noice_signal = np.fft.fft(Signal)
    
    if ch == 1:
        spect_signal = np.fft.fftshift(spect_signal)
        spect_noice_signal = np.fft.fftshift(spect_noice_signal)
    
        
    #Выводим графики
    t1 = np.arange(0,len(spect_signal))
    plt.figure(3,figsize=(13, 20))
    plt.subplot(4, 1, 1)
    plt.plot(t1, spect_signal, color='green')

    plt.xlabel('элемент массива')
    plt.ylabel('амплитуда')
    plt.title("Спектр передаваемого сигнала с N = 8")

    plt.subplot(4, 1, 2)
    plt.plot(t1, spect_noice_signal, color='brown')
    plt.xlabel('элемент массива')
    plt.ylabel('амплитуда')
    plt.title("Спектр принимаемого сигнала с шумом и N = 8")
    
    
    #13 задание выводим спект сигналов с разными N 
    samples_4N = np.repeat(exten_data_full,4)
    samples_16N = np.repeat(exten_data_full,16)
    Signal_4N = np.full(2*4*(exten_N + G), fill_value=float(0))
    Signal_16N = np.full(2*16*(exten_N + G), fill_value=float(0))
    for i in range(0, len(samples_4N)):
        Signal_4N[i]=samples_4N[i]
    for i in range(0, len(samples_16N)):
        Signal_16N[i]=samples_16N[i]
    
    spect_signal_4N = np.fft.fft(Signal_4N) 
    spect_signal_16N = np.fft.fft(Signal_16N)
        
    if ch == 1:
        spect_signal_4N = np.fft.fftshift(spect_signal_4N)
        spect_signal_16N = np.fft.fftshift(spect_signal_16N)
        
    noice = np.random.normal(0, q, len(Signal_4N))
    for i in range(len(Signal_4N)):
        Signal_4N[i] = Signal_4N[i] + noice[i]
    noice = np.random.normal(0, q, len(Signal_16N))
    for i in range(len(Signal_16N)):
        Signal_16N[i] = Signal_16N[i] + noice[i]

    spect_noice_signal_4N = np.fft.fft(Signal_4N)
    
    spect_noice_signal_16N = np.fft.fft(Signal_16N)
    
    if ch == 1:
        spect_noice_signal_4N = np.fft.fftshift(spect_noice_signal_4N)
        spect_noice_signal_16N = np.fft.fftshift(spect_noice_signal_16N)
    
    #Выводим все на график
    t2 = np.arange(0,len(spect_noice_signal_4N))
    t3 = np.arange(0,len(spect_noice_signal_16N))
    
    plt.figure(4,figsize=(13, 20))   
    plt.subplot(4, 1, 1)
    plt.plot(t3, spect_signal_16N, color='blue')
    plt.plot(t1, spect_signal, color='brown')
    plt.plot(t2, spect_signal_4N, color='green')
    plt.xlabel('элемент массива')
    plt.ylabel('амплитуда')
    plt.title("Спектр передаваемого сигнала с N = 8, N=4, N = 16")

    plt.subplot(4, 1, 2)
    plt.plot(t3, spect_noice_signal_16N, color='blue')
    plt.plot(t1, spect_noice_signal, color='brown')
    plt.plot(t2, spect_noice_signal_4N, color='green')



    plt.xlabel('элемент массива')
    plt.ylabel('амплитуда')
    plt.title("Спектр принимаемого сигнала с шумом и N = 8, N=4, N = 16")
        
