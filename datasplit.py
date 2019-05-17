import pandas as pd
import numpy as np

def split(dataframe,train_pct):
    '''
    this method splits the dataset in 6, 3 with train val test features and 3 with train val test labels
    returns x_train,x_val,x_test,y_train,y_val,y_test
    '''
    TRAIN_PCT = train_pct
    OTHER_PCT = 1-((1-TRAIN_PCT)/2)

    neutral_df =  dataframe[dataframe['new_emotion_cd']==0]
    happy_df =  dataframe[dataframe['new_emotion_cd']==1]
    surprise_df =  dataframe[dataframe['new_emotion_cd']==2]
    sad_df =  dataframe[dataframe['new_emotion_cd']==3]
    anger_df =  dataframe[dataframe['new_emotion_cd']==4]
    disgust_df =  dataframe[dataframe['new_emotion_cd']==5]
    fear_df =  dataframe[dataframe['new_emotion_cd']==6]
    contempt_df =  dataframe[dataframe['new_emotion_cd']==7]
    
    neutral_train, neutral_val, neutral_test = np.split(neutral_df.sample(frac=1), [int(TRAIN_PCT*len(neutral_df)), int(OTHER_PCT*len(neutral_df))])
    happy_train, happy_val, happy_test = np.split(happy_df.sample(frac=1), [int(TRAIN_PCT*len(happy_df)), int(OTHER_PCT*len(happy_df))])
    surprise_train, surprise_val, surprise_test = np.split(surprise_df.sample(frac=1), [int(TRAIN_PCT*len(surprise_df)), int(OTHER_PCT*len(surprise_df))])
    sad_train, sad_val, sad_test = np.split(sad_df.sample(frac=1), [int(TRAIN_PCT*len(sad_df)), int(OTHER_PCT*len(sad_df))])
    anger_train, anger_val, anger_test = np.split(anger_df.sample(frac=1), [int(TRAIN_PCT*len(anger_df)), int(OTHER_PCT*len(anger_df))])
    disgust_train, disgust_val, disgust_test = np.split(disgust_df.sample(frac=1), [int(TRAIN_PCT*len(disgust_df)), int(OTHER_PCT*len(disgust_df))])
    fear_train, fear_val, fear_test = np.split(fear_df.sample(frac=1), [int(TRAIN_PCT*len(fear_df)), int(OTHER_PCT*len(fear_df))])
    contempt_train, contempt_val, contempt_test = np.split(contempt_df.sample(frac=1), [int(TRAIN_PCT*len(contempt_df)), int(OTHER_PCT*len(contempt_df))])
    
    train_frames = [neutral_train,happy_train,surprise_train,sad_train,anger_train,disgust_train,fear_train,contempt_train]
    val_frames = [neutral_val,happy_val,surprise_val,sad_val,anger_val,disgust_val,fear_val,contempt_val]
    test_frames = [neutral_test,happy_test,surprise_test,sad_test,anger_test,disgust_test,fear_test,contempt_test]
    
    train = pd.concat(train_frames)
    val = pd.concat(val_frames)
    test = pd.concat(test_frames)
    
    train = train.sample(frac=1)
    val = val.sample(frac=1)
    test = test.sample(frac=1)
    
    x_train = train['pixels'] 
    x_val = val['pixels']
    x_test = test['pixels']
    
    y_train = train['new_emotion_cd']
    y_val = val['new_emotion_cd']
    y_test = test['new_emotion_cd']

    return x_train,x_val,x_test,y_train,y_val,y_test

#fer_dir = 'C:/Users/franco.ferrero/Documents/Datasets/fer2013-OZ.csv'
#TRAIN_PCT = 0.6
#
#
#fer_dataset = pd.read_csv(fer_dir)
#fer_dataset.drop(['old_emotion_cd','odl_emotion_desc','old_usage','new_emotion_desc'],axis=1,inplace=True)
##print('Old DS Head\n',fer_dataset.head(),'\n')
##print('Old DS Columns\n',fer_dataset.columns,'\n')
##print('Old emotion count\n', fer_dataset['old_emotion_cd'].value_counts())
##print('New emotion count\n', fer_dataset['new_emotion_cd'].value_counts())
##new_fer_ds= fer_dataset[['pixels','new_emotion_cd']]
##print('New DS Head\n',new_fer_ds.head(),'\n')
##print('New DS Columns\n',new_fer_ds.columns,'\n')
#
#
#
#
##print('neutral {}\nhappy {}\nsurprise {}\nsadness {}\nanger {}\ndisgust {}\nfear {}\ncontempt {}\n'.format(len(neutral_df)
##,len(happy_df)
##,len(surprise_df)
##,len(sad_df)
##,len(anger_df)
##,len(disgust_df)
##,len(fear_df)
##,len(contempt_df)))
#
#
#x_train,x_val,x_test,y_train,y_val,y_test = split(fer_dataset,TRAIN_PCT)
#
#print('x_train shape {}, x_val shape {}, x_test shape {}, y_train shape {}, y_val shape {}, y_test shape {}'.format(x_train.shape
#      ,x_val.shape,x_test.shape,y_train.shape,y_val.shape,y_test.shape))
#
#
##from sklearn import model_selection as ms
##
##df_train, df_val_test = ms.train_test_split(fer_dir,train_size=0.6,shuffle=True)#,stratify=fer_dir['new_emotion_cd'])
##df_val, df_test = ms.train_test_split(df_val_test, test_size=0.5,shuffle=True)#,stratify=df_val_test['new_emotion_cd'])
##
##print(df_test)
##print('orig shape {}, train shape {}, val shape{}, test shape{}'.format(neutral_df.shape,neutral_train.shape,neutral_val.shape,neutral_test.shape))