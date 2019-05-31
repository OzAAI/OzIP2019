import os
import pandas as pd
import numpy as np

def split(dataframe,train_pct):
    '''
    this method splits the dataset in 6, 3 with train val test features and 3 with train val test labels
    returns x_train,x_val,x_test,y_train,y_val,y_test
    '''
#    we get the train percentage and divide the remaining between validation and test equally
#    this work by setting % where to cut the data, for example, if we were to receive 0.6
#    we will set cuts at 0.6 and 0.8. So 0 to 0.6 is train, 0.6 to 0.8 is val and 0.8 to 1 is test
    TRAIN_PCT = train_pct
    OTHER_PCT = 1-((1-TRAIN_PCT)/2)
    
#    we want to keep the class percentages by dataset division. (for example if happy faces represent a 20% of our whole dataset
#    we would like to have 20% of train be happy faces, 20% of validation and 20% of test)
#    one way to do that is to split them up, divide them and then join them back again. So we are first going to split by emotion
    neutral_df =  dataframe[dataframe['new_emotion_cd']==0]
    happy_df =  dataframe[dataframe['new_emotion_cd']==1]
    surprise_df =  dataframe[dataframe['new_emotion_cd']==2]
    sad_df =  dataframe[dataframe['new_emotion_cd']==3]
    anger_df =  dataframe[dataframe['new_emotion_cd']==4]
    disgust_df =  dataframe[dataframe['new_emotion_cd']==5]
    fear_df =  dataframe[dataframe['new_emotion_cd']==6]
    contempt_df =  dataframe[dataframe['new_emotion_cd']==7]
    
#    now we will split each emotion in train/val/test according to our desired percentages
    neutral_train, neutral_val, neutral_test = np.split(neutral_df.sample(frac=1), [int(TRAIN_PCT*len(neutral_df)), int(OTHER_PCT*len(neutral_df))])
    happy_train, happy_val, happy_test = np.split(happy_df.sample(frac=1), [int(TRAIN_PCT*len(happy_df)), int(OTHER_PCT*len(happy_df))])
    surprise_train, surprise_val, surprise_test = np.split(surprise_df.sample(frac=1), [int(TRAIN_PCT*len(surprise_df)), int(OTHER_PCT*len(surprise_df))])
    sad_train, sad_val, sad_test = np.split(sad_df.sample(frac=1), [int(TRAIN_PCT*len(sad_df)), int(OTHER_PCT*len(sad_df))])
    anger_train, anger_val, anger_test = np.split(anger_df.sample(frac=1), [int(TRAIN_PCT*len(anger_df)), int(OTHER_PCT*len(anger_df))])
    disgust_train, disgust_val, disgust_test = np.split(disgust_df.sample(frac=1), [int(TRAIN_PCT*len(disgust_df)), int(OTHER_PCT*len(disgust_df))])
    fear_train, fear_val, fear_test = np.split(fear_df.sample(frac=1), [int(TRAIN_PCT*len(fear_df)), int(OTHER_PCT*len(fear_df))])
    contempt_train, contempt_val, contempt_test = np.split(contempt_df.sample(frac=1), [int(TRAIN_PCT*len(contempt_df)), int(OTHER_PCT*len(contempt_df))])
    
#    we don't want 24 dataframes, so we will now join all train,val and test emotions
    train_frames = [neutral_train,happy_train,surprise_train,sad_train,anger_train,disgust_train,fear_train,contempt_train]
    val_frames = [neutral_val,happy_val,surprise_val,sad_val,anger_val,disgust_val,fear_val,contempt_val]
    test_frames = [neutral_test,happy_test,surprise_test,sad_test,anger_test,disgust_test,fear_test,contempt_test]
    
    train = pd.concat(train_frames)
    val = pd.concat(val_frames)
    test = pd.concat(test_frames)
    
#    a simple shuffle so we get all emotions randomly mixed
    train = train.sample(frac=1)
    val = val.sample(frac=1)
    test = test.sample(frac=1)
    
#    lastly we divide into features and lables
    x_train = train['pixels'] 
    x_val = val['pixels']
    x_test = test['pixels']
    
    y_train = train[['new_emotion_cd','new_emotion_desc']]
    y_val = val[['new_emotion_cd','new_emotion_desc']]
    y_test = test[['new_emotion_cd','new_emotion_desc']]

    return x_train,x_val,x_test,y_train,y_val,y_test

def load_and_save(_read_dir_,_filename_,_write_dir_):
    #datasplit test
    fer_dataset = pd.read_csv(os.path.join(_read_dir_,_filename_))
    x_train,x_val,x_test,y_train,y_val,y_test = split(fer_dataset[['pixels','new_emotion_cd','new_emotion_desc']],0.6)
    print('train shape {}, val shape {}, test shape {}'.format(x_train.shape, x_val.shape, x_test.shape))
    
    x_train.to_csv(os.path.join(_write_dir_,r'x_train.csv'),header=True,index=False)
    x_val.to_csv(os.path.join(_write_dir_,r'x_val.csv'),header=True,index=False)
    x_test.to_csv(os.path.join(_write_dir_,r'x_test.csv'),header=True,index=False)
    y_train.to_csv(os.path.join(_write_dir_,r'y_train.csv'),header=True,index=False)
    y_val.to_csv(os.path.join(_write_dir_,r'y_val.csv'),header=True,index=False)
    y_test.to_csv(os.path.join(_write_dir_,r'y_test.csv'),header=True,index=False)