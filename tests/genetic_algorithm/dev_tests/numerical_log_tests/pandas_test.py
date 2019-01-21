import pandas as pd
import datetime


one = 1
two = 2

numeri_arabi = {'data1':one,'data2':two}
serie1= pd.Series(numeri_arabi,name='serie1')

one= 'I'
two= 'II'
numeri_romani = {'data1':one,'data2':two}
serie2= pd.Series(numeri_romani,name='serie2')


frame1 = pd.concat([serie1,serie2],axis=1)
frame2 = pd.concat([serie1,serie2],axis=0)

writer = pd.ExcelWriter('numerical_log.xlsx')
frame1.to_excel(writer,'Sheet1')
frame2.to_excel(writer,'Sheet2')

writer.save()

foglio1 = pd.read_excel('numerical_log.xlsx','Sheet1')
foglio2 = pd.read_excel('numerical_log.xlsx','Sheet2')

print('PRIMO: ',foglio1)
print('SECONDO: ',foglio2)

vuoto = pd.DataFrame()
tutto = pd.concat([vuoto,foglio1,foglio2])
print(tutto)


