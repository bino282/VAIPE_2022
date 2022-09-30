
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
def dict2json(dictionary, out_file='output.json'):
    try:
        with open(out_file, "w") as outfile:
            json.dump(dictionary, outfile)
        return True
    except:
        return False
    
class COCOeval_wmAP(COCOeval):
    def __init__(self, gt_coco, res_coco, iouType='segm', num_cls=75, alpha=10):
        super(COCOeval_wmAP, self).__init__(gt_coco, res_coco, iouType)
        self.num_cls = num_cls
        self.weights = np.array([1 if i!=num_cls-1 else alpha for i in range(num_cls)])


    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can only be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                s[s==-1] =0
                mean_s = np.mean(np.average(s, weights=self.weights, axis=-2))
                # mean_s = np.mean(s)
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        
        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats
        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()
        
def eval_wmap(pred_csv_path, data_dir = '/datahdd/public_test/pill/image/'):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import pandas as pd
    import cv2
    import os 
    
    pred_csv = pd.read_csv(pred_csv_path)
    coco_eval = {
        "images": [
            # {"id": 242287, "width": 426, "height": 640, "file_name": "xxxxxxxxx.jpg", "date_captured": "2013-11-15 02:41:42"},
        ],
        "annotations": [
            # {"category_id": 0, "image_id": 242287, "bbox": [19.23, 383.18, 314.5, 244.46]},
        ],
        "categories": [
            # {"id": 0,"name": "echo"},
            # {"id": 1,"name": "echo dot"}
        ]
    }

    preds = [
        # {"image_id":42,"category_id":18,"bbox":[258.15,41.29,348.26,243.78],"score":0.236}
    ]
    # add classes
    classes = [str(x) for x in range(108)]

    images2id = {}
    with open("datasets/gt.json","r",encoding="utf-8") as fr:
        gt = json.load(fr)
    images = []
    for e in gt["images"]:
        images.append(e["file_name"])
    images = list(set(images))
    for image_name in images:
        images2id[image_name] = image_name.split(".")[0]
    
    preds = []
    for i in range(len(pred_csv)):
        image_name = pred_csv.at[i, 'image_name']
        class_id = int(pred_csv.at[i, 'class_id'])
        confidence_score = float(pred_csv.at[i, 'confidence_score'])
        x_min = int(pred_csv.at[i, 'x_min'])
        y_min = int(pred_csv.at[i, 'y_min'])
        x_max = int(pred_csv.at[i, 'x_max'])
        y_max = int(pred_csv.at[i, 'y_max'])
        preds.append({
            "id": i,
            "category_id": class_id,
            "image_id": images2id[image_name],
            "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
            "score": round(confidence_score, 3),
        })
    dict2json(preds, 'datasets/pred.json')
    
    anno =COCO('datasets/gt.json')
    pred = anno.loadRes('datasets/pred.json')
    cocoEval = COCOeval_wmAP(anno, pred, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    map, map50 = cocoEval.stats[:2]
    return map, map50

if __name__ == "__main__":
    pred_path = ROOT + "output_results/beit/results_old.csv"
    map, map50 = eval_wmap(pred_path)
    print(map, map50)