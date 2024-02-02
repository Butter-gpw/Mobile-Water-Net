from Under_Water_index import Under_Water_index_Proceesing


tested_path=''  # path to tested images folder
ref_path = '' # path to reference images folder

uwip = Under_Water_index_Proceesing(tested_path,ref_path, img_size=(512,512))

uwip.calculate()

uwip.form.to_csv('result.csv',index=False)
uwip.form.describe().to_csv('result_describe.csv')