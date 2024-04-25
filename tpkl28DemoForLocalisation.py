import gradio as gr
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
import timm
from torchvision import datasets, models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam import EigenGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import torchvision
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

    elif flag == 'Swin-Small':
        EfficientNetTransforms = transforms.Compose([transforms.Resize((224, 224)),
                                                     # transforms.RandomHorizontalFlip(),
                                                     # transforms.RandomRotation(degrees=15),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                          std=[0.229, 0.224, 0.225])

                                                     ]
                                                    )
        ModelTransforms = EfficientNetTransforms
        CheckPoint = torch.load(f'Models/Swin-Small-BEST.pt')

        model_name = 'swin_small_patch4_window7_224.ms_in22k'
        model = timm.create_model(model_name, pretrained=False, num_classes=NumClasses)
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
def CAMMethods(model,image,Threshold=0.5,label=False):
    ##A function to take any model and generate class activation maps with parameters
    #Extend for eigen ablation and other versions of Grad-CAM
    classEncoding = {
        'Atelectasis': torch.FloatTensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Consolidation': torch.FloatTensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Infiltration': torch.FloatTensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Pneumothorax': torch.FloatTensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Edema': torch.FloatTensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Emphysema': torch.FloatTensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Fibrosis': torch.FloatTensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Effusion': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
        'Pneumonia': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
        'Pleural_Thickening': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
        'Cardiomegaly': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
        'Nodule': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
        'Hernia': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
        'Mass': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
        'No Finding': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    }
    #SWIN Reshape !
    def reshape_transform(tensor,height=7,width=7):
        #result = tensor.reshape(tensor.size(0),height,width,tensor.size(3))
        result = tensor.transpose(2,3).transpose(1,2)
        return result
    ##For VIT
    model.eval()
    model.to(device)

    #target_layers = [model.layers[-1].blocks[-1].norm1,model.layers[-1].blocks[-2].norm1]
    #target_layers = [model.conv_head]

    target_layers = [model.features[-1]]
    #cam = EigenGradCAM(model=model,target_layers=target_layers,reshape_transform=reshape_transform)
    cam = EigenGradCAM(model=model, target_layers=target_layers)
    cam.batch_size = 1

    targets = label
    #targets = [ClassifierOutputTarget(0)]
    #Change for different pathologies
    #targets = [classEncoding['Atelectasis']]

    #targets = [ClassifierOutputTarget(0)]
    #targets = 'Test'
    #targets = None
    #greyscaleCam.to(device)
    #targets = 10
    greyscaleCam = cam(input_tensor=image,targets=targets,aug_smooth=False)
    #display_tensor_as_image(greyscaleCam[0,:])
    greyscaleCam = greyscaleCam[0,:]
    for x in range(len(greyscaleCam)):
        for y in range(len(greyscaleCam[0])):
            item = greyscaleCam[x][y]
            if item < Threshold:
                greyscaleCam[x][y] = 0
    RGBImage = torchvision.transforms.functional.to_pil_image(image.squeeze(0))
    #visualisation = show_cam_on_image(RGBImage,greyscaleCam)
    return greyscaleCam
def ApplyColourMap(GradCam):
    GradCam = (GradCam - np.min(GradCam) / (np.max(GradCam) - np.min(GradCam)))

    colourMap = plt.get_cmap('jet')
    heatMap = colourMap(GradCam)[:,:,:3]
    heatMap = (heatMap * 255).astype(np.uint8)
    heatMap = Image.fromarray(heatMap)
    return heatMap
def ShowGradCam(image,GradCam):

    return Image.blend(image.resize([224, 224]), ApplyColourMap(GradCam), alpha=0.3)
def GenerateOverlayImage(model,image,LabelNumber):
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##')
    optimalThrehsolds = {0: (0.26621786, 0.4249210700615868), 1: (0.09752839, 0.2545985888343932),
                         2: (0.21465598, 0.35761881931528816), 3: (0.17483784, 0.32599070319482737),
                         4: (0.15293084, 0.2622945843384878), 5: (0.19395962, 0.35830569699480797),
                         6: (0.0425542, 0.1425887041608142), 7: (0.2941757, 0.577877612827387),
                         8: (0.02460439, 0.09036117747575677), 9: (0.1547481, 0.22713816378308774),
                         10: (0.1548717, 0.325410856652741), 11: (0.1138406, 0.2878146276600149),
                         12: (0.20437676, 0.49999958680589707), 13: (0.14593615, 0.42986375365017865)}
    ListedThresholds = [i[0] for i in list(optimalThrehsolds.values())]
    modelPackage = LoadModel(model)
    model = modelPackage[0]
    transforms = modelPackage[1]
    ImageTensor = transforms(image)
    ImageTensor.to(device)
    ImageTensor = ImageTensor.unsqueeze(0)
    prediction = model(ImageTensor)
    prediction = torch.sigmoid(prediction).detach().numpy()
    #print(prediction)
    predicted_labels = (prediction > ListedThresholds).astype(int)
    labels = list(classEncoding.keys())

    result = {label: (1.0 if label_status else 0.0) for label, label_status in zip(labels, predicted_labels[0])}
    for item in result.keys():
        if result[item] == 1.0:
            print(item)
    #LabelNumber = classEncoding[LabelNumber]
    LabelNumber = (classEncoding[LabelNumber].argmax(dim=0))
    Map = CAMMethods(model,ImageTensor,0.25,[ClassifierOutputTarget(LabelNumber)])
    ReturnImage = ShowGradCam(image,Map)
    return ReturnImage
#CurrentImage = Image.open(CurrentFileName)
#CurrentImage = CurrentImage.convert('RGB')

#Map = CAMMethods(self.model,ImageTensor,0.1,[ClassifierOutputTarget(LabelNumber)])
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
LabelsDrop = gr.Dropdown(list(classEncoding.keys()))
ModelsDrop = gr.Dropdown(['DenseNet-121','ENet-b1','ENet-b5','ViT-Base','MbNv2','Swin-Small'])
iface = gr.Interface(fn=GenerateOverlayImage,
                     inputs=[ModelsDrop,gr.Image(type='pil'),LabelsDrop],
                     outputs =gr.Image(type='pil'),

                     title = 'Localisation Demo',
                     description='Select a model,image and target prediction and go !')
iface.launch()