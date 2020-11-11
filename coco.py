import json


captions={}
instances={}
fw1=open('captions_noid.txt','w',encoding='utf')
fw2=open('instances_noid.txt','w',encoding='utf')

with open("captions_train2014.json",'r') as load_f:
    data= json.load(load_f)
    print(data.keys())
    print('len annotations'+str(len(data['annotations'])))
    for i,anno in enumerate(data['annotations']):
        # if i<10:
        #     print(anno)
        if anno['image_id'] not in captions.keys():
            captions[anno['image_id']]=anno['caption']
    print('len captions:'+str(len(captions)))


class_map={}
with open("instances_train2014.json",'r') as load_f:
    data= json.load(load_f)
    # print(data.keys())
    class_list=data['categories']
    for cl in class_list:
        class_map[cl['id']]=cl['name']
    # print(class_map)

    print('len annotations:'+str(len(data['annotations'])))
    for i,anno in enumerate(data['annotations']):
        # if i<2:
        #     print(anno)
        if anno['image_id'] not in instances.keys():
            instances[anno['image_id']]= [class_map[anno['category_id']]]
        else:
            instances[anno['image_id']] += [class_map[anno['category_id']]]
    print('len instances'+str(len(instances)))

set1=captions.keys()
set2=instances.keys()
inter_set=set1&set2
print('交集'+str(len(inter_set)))
d_order = sorted(captions.items(), key=lambda x: x[0], reverse=False)
for id_cp in d_order:
    try:
        id=id_cp[0]
        if id in inter_set:
            fw1.write(id_cp[1] + '\n')
    except:
        print(id_cp)
d_order=sorted(instances.items(),key=lambda x:x[0],reverse=False)
for id_cp in d_order:
    try:
        id=id_cp[0]
        if id in inter_set:
            fw2.write(' '.join(id_cp[1]) + '\n')
    except:
        # pass
        print(id_cp)
fw1.close()
fw2.close()


