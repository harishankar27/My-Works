from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import load_model
from preprocess import *

# Load the best model saved during training
best_model = load_model('best_model.h5')

predictions = best_model.predict(test_data)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = n_labels_test

confusion_mat = confusion_matrix(true_classes, predicted_classes)
print("Confusion Matrix:")
print(confusion_mat)

import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix

def performance_matrix(true, pred):
    precision = metrics.precision_score(true,pred,average='weighted')
    recall = metrics.recall_score(true,pred,average='weighted') # average='weighted'
    accuracy = metrics.accuracy_score(true,pred)
    f1_score = metrics.f1_score(true,pred,average='weighted')
    print('Mean \n  precision: {} \n  recall: {}, \n  accuracy: {}, \n  f1_score: {}'.format(precision*100,recall*100,accuracy*100,f1_score*100))

performance_matrix(true_classes, predicted_classes)

# 
from tabulate import tabulate
from sklearn.metrics import precision_recall_fscore_support as score
precision, recall, fscore, support = score(true_classes, predicted_classes)

print('\n')
print('rows is precision, recall, fscore and support:')
print(tabulate([precision, recall, fscore, support], headers=['0' , '1' , '2' , '3', '4'], tablefmt='orgtbl'))

print('\n')
print('per-class accuracy:')
cm = confusion_matrix(true_classes, predicted_classes)
cm = cm.astype('float')*100 / cm.sum(axis=1)[:, np.newaxis]
print(tabulate([cm.diagonal()], headers=['0' , '1' , '2' , '3', '4'], tablefmt='orgtbl'))