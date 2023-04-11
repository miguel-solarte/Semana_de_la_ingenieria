from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from captura_imagen import deteccion

if __name__ == '__main__':

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.2).to('cuda')
    preprocess = weights.transforms()
    model.eval()


    #path = '../../../../mnt/g/Mi unidad/video_20230329_114555.mp4'
    path = './video_20230329_114555.mp4'
    color_boxes = (0, 0, 255) # R G B

    deteccion(path,model,preprocess,color_boxes)