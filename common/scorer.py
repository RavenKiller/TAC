# -*- coding=utf-8 -*-
# author: w61, raven
# Test for several ways to compute the score of the generated words.
from .coco_caption.pycocoevalcap.bleu.bleu import Bleu
from .coco_caption.pycocoevalcap.meteor.meteor import Meteor
from .coco_caption.pycocoevalcap.rouge.rouge import Rouge
from .coco_caption.pycocoevalcap.cider.cider import Cider
from .coco_caption.pycocoevalcap.spice.spice import Spice
from .coco_caption.pycocoevalcap.wmd.wmd import WMD


class Scorer:
    def __init__(self):
        self.scorers = [
            (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE"),
            (WMD(), "WMD"),
        ]

    def compute_scores(self, ref, gt):
        total_scores = {}
        for scorer, method in self.scorers:
            # print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gt, ref)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    # scs保存了各个单样本
                    # sc是全体样本算的
                    # closest选项控制reflen的选择
                    # print(m, sc, scs)
                    total_scores[m] = sc
            else:
                # print("%s: %0.3f"%(method, score))
                total_scores[method] = score

        # print('*****DONE*****')
        # for key,value in total_scores.items():
        #     print('{}:{}'.format(key,value))
        return total_scores


if __name__ == "__main__":
    ref = {
        "1": ["go down the stairs and stop at the bottom ."],
        "2": ["this is a cat."],
    }
    gt = {
        "1": [
            "Walk down the steps and stop at the bottom. ",
            "Go down the stairs and wait at the bottom.",
            "Once at the top of the stairway, walk down the spiral staircase all the way to the bottom floor. Once you have left the stairs you are in a foyer and that indicates you are at your destination.",
        ],
        "2": ["It is a cat.", "There is a cat over there.", "cat over there."],
    }
    # 注意，这里如果只有一个sample，cider算出来会是0，详情请看评论区。
    scorer = Scorer()
    res = scorer.compute_scores(ref, gt)
    print(res)
