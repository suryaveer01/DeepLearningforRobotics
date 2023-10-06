from ultralytics import YOLO

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



from CustomDataset import CustomDataset


dataset = CustomDataset(root_dir='Datasets/imagenet_images')

print('len(dataset))',len(dataset))


class_to_idx = dataset.class_to_idx 

idx_to_class = {}
conf_matrix_labels = ['Unrecognized']

for key, value in class_to_idx.items():
    idx_to_class[value] = key
    conf_matrix_labels.append(key)
    
print('idx_to_class',idx_to_class)
# conf_matrix_labels = (list(idx_to_class.values())).append('Unrecognized')
print('conf_matrix_labels',conf_matrix_labels)

model = YOLO('yolov8x-cls.pt')
# image_pth = 'Datasets/imagenet_images/seashore/58649744_06b8bd8c56.jpg'

# results = model(image_pth)  # predict on an image

correct_top_1 = 0
correct_top_5 = 0
total = 0
test_loss = 0.0
top_1_all_preds = []
top_5_all_preds = []
all_labels = []
num_classes = 4

for img,label in dataset:
    # print(label)
    results = model(img)  # predict on an image and classify
# results = model.pred
# print('results.names: ',results[0].names)
    
    result = results[0]
    # print('top1 : ',result.probs.top1)
    # print('confidence:',result.probs.top1conf)
    # print('top5 : ',result.probs.top5)
    # print('confidence:',result.probs.top5conf)
    # print('class name:',result.names[result.probs.top1])
    # print('class name top5:',[result.names[name] for name in result.probs.top5])
    top_1_result = result.names[result.probs.top1]
    top_5_results = [result.names[name] for name in result.probs.top5]

    all_labels.append(idx_to_class[label])

    if( top_1_result == idx_to_class[label]):
        correct_top_1 += 1
        top_1_all_preds.append(top_1_result)
    else:
        top_1_all_preds.append('Unrecognized')
    # top_1_all_preds.append(class_to_idx.get(result.names[result.probs.top1], -1))

    if( idx_to_class[label] in top_5_results):
        correct_top_5 += 1
        top_5_all_preds.append(idx_to_class[label])
    else:
        top_5_all_preds.append('Unrecognized')


    # top_5_all_preds.append([class_to_idx.get(name, -1) for name in result.probs.top5])
    total += 1
    # if result.names[result.probs.top1] == idx_to_class[label]:
    #     correct_top_1 += 1
    # if idx_to_class[label] in top_5_results:
    #     correct_top_5 += 1

test_acc_top1 = correct_top_1 / total

test_acc_top5 = correct_top_5 / total

# Log Testing accuracy and loss to TensorBoard
print('Alllabelslen ',all_labels[0])
print('top_1_all_preds len ',top_1_all_preds[0])
print('top_5_all_preds len ',top_5_all_preds[0])


print(f'Test Accuracy top1: {test_acc_top1:.4f} Test Accuracy top5: {correct_top_5 / total:.4f}')


conf_matrix_top_1 = confusion_matrix(all_labels, top_1_all_preds)

cmd_top1 = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_top_1, display_labels=conf_matrix_labels)
cmd_top1.plot(cmap='Blues',colorbar=False)
cmd_top1.ax_.set(xlabel='Predicted', ylabel='True')
cmd_top1.figure_.savefig('YOlO_confusion_matrix_top1.png')
# cmd_top1.savefig('YOlO_confusion_matrix_top1.png')

# plt.figure(figsize=(num_classes, num_classes))
# sns.heatmap(conf_matrix_top_1, annot=True, fmt='d', cmap='Blues', cbar=False)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# # plt.xticks(range(num_classes), class_to_idx.keys())
# # plt.yticks(range(num_classes), class_to_idx.keys())
# plt.xticks([''] + conf_matrix_labels)
# plt.yticks([''] + conf_matrix_labels)
# plt.title('Confusion Matrix Top1')
# # plt.show()
# plt.savefig('YOlO_confusion_matrix_top1.png')
# plt.close()


conf_matrix_top_5 = confusion_matrix(all_labels, top_5_all_preds)

cmd_top5 = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_top_5, display_labels=conf_matrix_labels,)
cmd_top5.plot(cmap='Blues',colorbar=False)
cmd_top5.ax_.set(xlabel='Predicted', ylabel='True')
cmd_top5.figure_.savefig('YOlO_confusion_matrix_top5.png')

# plt.figure(figsize=(num_classes, num_classes))
# sns.heatmap(conf_matrix_top_5, annot=True, fmt='d', cmap='Blues', cbar=False)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# # plt.xticks(range(num_classes), class_to_idx.keys())
# # plt.yticks(range(num_classes), class_to_idx.keys())
# plt.xticks([''] + conf_matrix_labels)
# plt.yticks([''] + conf_matrix_labels)
# plt.title('Confusion Matrix Top5')
# # plt.show()
# plt.savefig('YOlO_confusion_matrix_top5.png')
# plt.close()


        
    

    
# model.train(data = 'Datasets/imagenet_images/',epochs=1)
# trainer = engine.trainer.BaseTrainer()
# print(trainer)
    