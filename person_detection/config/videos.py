import os

video_root = '/home/wzt/Videos/PFD/'

local_videos = [
    "bst.mp4", # 0
    "example_01.mp4", # 1
    "example_02.mp4", # 2
    "street.mp4", # 3
    "TownCentreXVID.avi", # 4
    "【小崔居家脱口秀】美国69%会出门戴口罩| 看小崔如何逻辑输出不守规则、不戴口罩的自由主义者 - 001 - Why Are We Still Having The Mask Debate  The Daily Social Distancing Show.mp4", # 5
    "这是什么神仙嗓音！口罩姐姐街头演唱千千阙歌，一开口就直击心灵 - 001 - wsy 这又是什么神仙嗓音 口罩姐姐街头演唱千千阙歌 一开口就直击心灵是注入灵魂.mp4", # 6
    "《红色高跟鞋》疯狂却怕没有退路 - 001 - 《红色高跟鞋》.mp4", # 7
    "长沙街唱，这位小姐姐的《水星记》真让人念念不忘！ - 001 - 长沙街唱，这位小姐姐的《水星记》真让人念念不忘！.mp4", # 8
    "长沙街唱《夏天的风》小姐姐人美歌甜！清新的感觉！ - 001 - 7.10 街唱 夏天的风.mp4", #9
    "不会真有人能街头唱《好想爱这个世界啊》吧？？！ - 001 - 不会真有人能街头唱《好想爱这个世界啊》吧？？！.mp4", #10
    "各国街头一分钟内能遇到几个不戴口罩的？ - 001 - 63.mp4", # 11
    "standard.flv", # 12
    "190111_07_SkywalkMahanakhon_HD_04.mp4", # 13
    "200910_02_Oxford_4K_021.mp4", # 14
]

rtsps = [
    "rtsp://192.168.0.121:8554/live", # 0
]

def select_local(idx: int) -> str:
    return os.path.join(video_root, local_videos[idx])

def select_rtsp(idx: int) -> str:
    return rtsps[idx]