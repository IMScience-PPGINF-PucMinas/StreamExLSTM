import os

for split_id in range(5):
    os.system('python -m model.main --split_index ' + str(split_id))
    os.system('python -m model.main --video_type TVSum --split_index ' + str(split_id))

print('Inference results for TVSum Dataset')
os.system('python -m inference.inference --dataset TVSum --test_dataset TVSum')
print('Inference results for TVSum Dataset')
os.system('python -m inference.inference')