from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import pickle
import gzip
#from sklearn import datasets
#iris = datasets.load_iris()
#print iris.target

def main():
    RGBData = []
    RGBBool = []
    IRData = []
    IRBool = []
    
    f = gzip.open( "0003.mp4SVMDATARGB.pklz", "rb" )
    RGBData.extend(pickle.load(f))
    f.close()
    f = gzip.open( "0003.mp4SVMBOOLSRGB.pklz", "rb" )
    RGBBool.extend(pickle.load(f))
    f.close()
    f = gzip.open( "0003.mp4SVMDATAIR.pklz", "rb" )
    IRData.extend(pickle.load(f))
    f.close()
    f = gzip.open( "0003.mp4SVMBOOLSIR.pklz", "rb" )
    IRBool.extend(pickle.load(f))
    f.close()
    
    f = gzip.open( "0004.mp4SVMDATARGB.pklz", "rb" )
    RGBData.extend(pickle.load(f))
    f.close()
    f = gzip.open( "0004.mp4SVMBOOLSRGB.pklz", "rb" )
    RGBBool.extend(pickle.load(f))
    f.close()
    f = gzip.open( "0004.mp4SVMDATAIR.pklz", "rb" )
    IRData.extend(pickle.load(f))
    f.close()
    f = gzip.open( "0004.mp4SVMBOOLSIR.pklz", "rb" )
    IRBool.extend(pickle.load(f))
    f.close()
    
    target_names = ["false","true"]      
    
    print("RGB results")        
    print("========================================================================================")
    
    clfRGB = svm.SVC()
    data_train,data_test,target_train,target_test=train_test_split(RGBData,RGBBool,test_size=0.3)
    print("Size of Training Data = %d\n" % len(data_train))
    print(clfRGB.fit(data_train, target_train ))
    
    predicted = clfRGB.predict(data_test)
    print(predicted)
    print(metrics.classification_report(target_test,predicted,target_names=target_names))
    print(metrics.confusion_matrix(target_test,predicted))
    print(metrics.roc_curve(target_test,predicted))
    f = open( "svmModelRGB.p", "wb")
    pickle.dump(clfRGB, f)
    f.close()

    print("IR results")
    print("========================================================================================")
    
    clfIR = svm.SVC()
    data_train,data_test,target_train,target_test=train_test_split(IRData,IRBool,test_size=0.3)
    print("Size of Training Data = %d\n" % len(data_train))
    print(clfIR.fit(data_train, target_train))
    predicted = clfRGB.predict(data_test)
    print(metrics.classification_report(target_test,predicted,target_names=target_names))
    print(metrics.confusion_matrix(target_test,predicted))
    print(metrics.roc_curve(target_test,predicted))
    f = open( "svmModelIR.p", "wb" )
    pickle.dump(clfRGB, f )
    f.close()
    
    

main()