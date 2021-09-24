from konlpy.tag import komoran

komoran = Komoran(userdic='/.user_dic.tsv')
text = "우리 챗봇은 엔엘피를 좋아해. 이종명은 안우기를 시샵해 바보야."
pos = komoran.pos(text)
print(pos)