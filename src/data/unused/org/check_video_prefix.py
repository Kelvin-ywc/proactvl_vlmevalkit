import json


ann_path = '/home/v-weicaiyan/ds/DATA/game_commentary/Minecraft/raw/convert_minecraft.json'
ann_save_path = '/home/v-weicaiyan/ds/DATA/game_commentary/Minecraft/raw/annotation_minecraft.json'
with open(ann_path, 'r') as f:
    anns = json.load(f)
for ann in anns:
    ann['video_path'] = ann['video_path'].split('.')[0] + '.mp4'
anns.sort(key=lambda x: int(x['video_path'][13:-4]))  
print(len(anns))
# 重新保存ann
with open(ann_save_path, 'w') as f:
    json.dump(anns, f, indent=4)
