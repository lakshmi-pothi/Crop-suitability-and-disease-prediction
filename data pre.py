import pandas as pd
crop_data_path = 'C:/Users/dcalv/OneDrive/Desktop/SRM/Major Project/PROJECT/app/Data-raw/cpdata.csv'
fertilizer_data_path = 'C:/Users/dcalv/OneDrive/Desktop/SRM/Major Project/PROJECT/app/Data-raw/Fertilizer.csv'

crop = pd.read_csv(crop_data_path)
fert = pd.read_csv(fertilizer_data_path)
print(crop.head())
print(fert.head())
# Function for lowering the cases
def change_case(i):
    i = i.replace(" ", "")
    i = i.lower()
    return i
fert['Crop'] = fert['Crop'].apply(change_case)
crop['label'] = crop['label'].apply(change_case)

fert['Crop'] = fert['Crop'].replace('mungbeans','mungbean')
fert['Crop'] = fert['Crop'].replace('lentils(masoordal)','lentil')
fert['Crop'] = fert['Crop'].replace('pigeonpeas(toordal)','pigeonpeas')
fert['Crop'] = fert['Crop'].replace('mothbean(matki)','mothbeans')
fert['Crop'] = fert['Crop'].replace('chickpeas(channa)','chickpea')
print(crop.head())
print(crop.tail())
crop_names = crop['label'].unique()
print(crop_names)
print(fert.head())
del fert['Unnamed: 0']
crop_names_from_fert = fert['Crop'].unique()
print(crop_names_from_fert)
for i in crop_names_from_fert:
    print(crop[crop['label'] == i])
print(crop['label'])
extract_labels = []
for i in crop_names_from_fert:
    if i in crop_names:
        extract_labels.append(i)
# using extract labesl on crop to get all the data related to those labels
new_crop = pd.DataFrame(columns = crop.columns)
new_fert = pd.DataFrame(columns = fert.columns)
for label in extract_labels:
    new_crop = new_crop.append(crop[crop['label'] == label])
for label in extract_labels:
    new_fert = new_fert.append(fert[fert['Crop'] == label].iloc[0])
print(new_crop)
print(new_fert)
new_crop.to_csv('MergeFileCrop.csv')
new_fert.to_csv('FertilizerData.csv')