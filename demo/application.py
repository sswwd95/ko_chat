"""
@auther Hyunwoong
@since 7/1/2020
@see https://github.com/gusdnd852
"""

from flask import render_template

from kochat.app import KochatApi
from kochat.data import Dataset
from kochat.loss import CRFLoss, CosFace, CenterLoss, COCOLoss, CrossEntropyLoss
from kochat.model import intent, embed, entity
from kochat.proc import DistanceClassifier, GensimEmbedder, EntityRecognizer, SoftmaxClassifier
from datetime import datetime
import matplotlib
matplotlib.use('agg')

# from demo.scenario import dust, weather, travel, restaurant
from scenario import dust, weather, travel, restaurant
# 에러 나면 이걸로 실행해보세요!

start = datetime.now()
print("시이이이ㅣ이이ㅣ자아악!!!!!!!:", start)

dataset = Dataset(ood=True)
emb = GensimEmbedder(model=embed.FastText())

clf = DistanceClassifier(
    model=intent.CNN(dataset.intent_dict),
    loss=CenterLoss(dataset.intent_dict),
)

rcn = EntityRecognizer(
    model=entity.LSTM(dataset.entity_dict),
    loss=CRFLoss(dataset.entity_dict)
)

kochat = KochatApi(
    dataset=dataset,
    embed_processor=(emb, True),
    intent_classifier=(clf, True),
    entity_recognizer=(rcn, True),
    scenarios=[
        weather, dust, travel, restaurant
    ]
)


@kochat.app.route('/')
def index():
    return render_template("index.html")


if __name__ == '__main__':
    kochat.app.template_folder = kochat.root_dir + 'templates'
    kochat.app.static_folder = kochat.root_dir + 'static'
    kochat.app.run(port=8080, host='0.0.0.0')
    

end = datetime.now()
print("끝!!!!!!!:", end)
print('시간 : ', end - start)





#  * Running on http://192.168.0.60:8080/ (Press CTRL+C to quit)

#  2021-07-13 10:50:44.802471