import gradio as gr
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
import timm
from torchvision import datasets, models, transforms
device = torch.device('cuda')
def LoadModel(flag):
    NumClasses=14

    if flag == 'ViT-Base':
        model_name = 'vit_base_patch16_clip_224.openai_ft_in12k_in1k'
        model = timm.create_model(model_name,pretrained=False,num_classes=NumClasses)

        TempTransforms = transforms.Compose([transforms.Resize((224, 224)),
                                             #transforms.RandomHorizontalFlip(),

                                             #transforms.RandomRotation(degrees=15),
                                             # transforms.RandomResizedCrop()
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                                  std=[0.26862954, 0.26130258, 0.27577711])

                                             ])
        ModelTransforms = TempTransforms
        CheckPoint = torch.load(f'Models/ViT-Base-BEST.pt')


        model.load_state_dict(CheckPoint['state_dict'])


    elif flag == 'MbNv2':
        MobileNetV2Transforms = transforms.Compose([transforms.Resize((224, 224)),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomRotation(degrees=15),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225])

                                                    ]
                                                   )

        ModelTransforms = MobileNetV2Transforms
        CheckPoint = torch.load(f'Models/MobileNetV2-BEST.pt')
        model = models.mobilenet_v2(pretrained=True)


        model.classifier[1] = nn.Linear(model.last_channel, 14)
        model.load_state_dict(CheckPoint['state_dict'])



    elif flag == 'ENet-b5':
        EfficientNetTransforms = transforms.Compose([transforms.Resize((224, 224)),
                                                     #transforms.RandomHorizontalFlip(),
                                                     #transforms.RandomRotation(degrees=15),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                          std=[0.229, 0.224, 0.225])

                                                     ]
                                                    )
        ModelTransforms = EfficientNetTransforms
        CheckPoint = torch.load(f'Models/ENet-b5-BEST.pt')
        model_name = 'efficientnet_b5'
        model = timm.create_model(model_name, pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, NumClasses)
        model.load_state_dict(CheckPoint['state_dict'])

    elif flag == 'ENet-b1':
        EfficientNetTransforms = transforms.Compose([transforms.Resize((224, 224)),
                                                     # transforms.RandomHorizontalFlip(),
                                                     # transforms.RandomRotation(degrees=15),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                          std=[0.229, 0.224, 0.225])

                                                     ]
                                                    )
        ModelTransforms = EfficientNetTransforms
        CheckPoint = torch.load(f'Models/ENet-b1-BEST.pt')
        model_name = 'efficientnet_b1'
        model = timm.create_model(model_name, pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, NumClasses)
        model.load_state_dict(CheckPoint['state_dict'])
    elif flag == 'DenseNet-121':
        TempTransforms = transforms.Compose([transforms.Resize((224, 224)),
                                             #transforms.RandomHorizontalFlip(),
                                             #transforms.RandomRotation(degrees=15),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        ModelTransforms = TempTransforms
        CheckPoint = torch.load(f'Models/DenseNet-121-BEST.pt')
        model_name = 'densenet121.tv_in1k'
        model = timm.create_model(model_name, pretrained=True, num_classes=NumClasses)
        model.load_state_dict(CheckPoint['state_dict'])
    elif flag == 'DenseNet-169':
        CheckPoint = torch.load(f'Models/DenseNet-169-BEST.pt')
        model_name = 'densenet169.tv_in1k'
        model = timm.create_model(model_name, pretrained=True, num_classes=NumClasses)
        model.load_state_dict(CheckPoint['state_dict'])
    elif flag == 'DenseNet-201':
        CheckPoint = torch.load(f'Models/DenseNet-201-BEST.pt')
        model_name = 'densenet201.tv_in1k'
        model = timm.create_model(model_name, pretrained=False, num_classes=NumClasses)
        model.load_state_dict(CheckPoint['state_dict'])
    model.eval()
    return (model,ModelTransforms)
classEncoding = {
            'Atelectasis': torch.FloatTensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'Consolidation': torch.FloatTensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'Infiltration': torch.FloatTensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'Pneumothorax': torch.FloatTensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'Edema': torch.FloatTensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'Emphysema': torch.FloatTensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
            'Fibrosis': torch.FloatTensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
            'Effusion': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
            'Pneumonia': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
            'Pleural_Thickening': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
            'Cardiomegaly': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
            'Nodule': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
            'Hernia': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
            'Mass': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
            # 'No Finding': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        }
def ClassifyImage(model_name,image):
    modelPackage = LoadModel(model_name)
    model = modelPackage[0]
    Transforms = modelPackage[1]
    #image = Image.open('Data/images_001/images/00016618_006.png').convert('RGB')
    ImageForModel = Transforms(image).unsqueeze(0)

    optimalThrehsolds = {0: (0.26621786, 0.4249210700615868), 1: (0.09752839, 0.2545985888343932), 2: (0.21465598, 0.35761881931528816), 3: (0.17483784, 0.32599070319482737), 4: (0.15293084, 0.2622945843384878), 5: (0.19395962, 0.35830569699480797), 6: (0.0425542, 0.1425887041608142), 7: (0.2941757, 0.577877612827387), 8: (0.02460439, 0.09036117747575677), 9: (0.1547481, 0.22713816378308774), 10: (0.1548717, 0.325410856652741), 11: (0.1138406, 0.2878146276600149), 12: (0.20437676, 0.49999958680589707), 13: (0.14593615, 0.42986375365017865)}
    ListedThresholds = [i[0] for i in list(optimalThrehsolds.values())]
    with torch.no_grad():

        prediction = model(ImageForModel)
        prediction = torch.sigmoid(prediction).numpy()
        print(prediction)
        predicted_labels = (prediction > ListedThresholds).astype(int)
        labels = list(classEncoding.keys())
        result = {label: (label_status) for label, label_status in zip(labels, prediction[0])}
        #result = {prediction}
        return result
        #SoftmaxTensor = torch.sigmoid(prediction)
        #SoftmaxTensor = nn.functional.sigmoid(prediction)
        #print(SoftmaxTensor)
        #Outputs = {list(classEncoding.keys())[i]: float(prediction[0][i]) for i in range(14)}
    #print(Outputs)
    #heatmap = grad_cam(model,image)

    #return Outputs

ModelsDrop = gr.Dropdown(['DenseNet-121','ENet-b1','ENet-b5','ViT-Base','MbNv2'])
iface = gr.Interface(fn=ClassifyImage,
                     inputs=[ModelsDrop,gr.Image(type='pil')],
                     outputs =gr.Label(num_top_classes=14),

                     title = 'JcrawleyDemo',
                     description='Select a model and chose an image to classify')

iface.launch()