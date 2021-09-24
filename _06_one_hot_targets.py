from konlpy.tag import Komoran
import numpy as np

komoran = Komoran()
text = "오늘 날씨는 구름이 많아요."

# 명사만 추출
nouns = komoran.nouns(text)
print(nouns)

# 단어 사전 구축 및 단어별 인덱스 부여
dics = {}
for word in nouns:
    if word not in dics.keys():
        dics[word] = len(dics)
print(dics)

# 원 핫 인코딩
nb_classes = len(dics)
targets = list(dics.values())
# targets의 순서를 그대로 유지하라
one_hot_targets = np.eye(nb_classes)[targets]
print(one_hot_targets)

print(np.eye(nb_classes),end="\n\n")

print(np.eye(nb_classes)[[5,4,3,2,1,0]],end="\n\n")

print(np.eye(nb_classes)[[5,4,3,2,1,0]],end="\n\n")