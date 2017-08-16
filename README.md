# cuhk03
pytorch train on cuhk03 dataset

test with 100 images from camera a in probe(one person one image), 100 images from camera b in gallery

1. classification.py:
   classification loss, use CrossEntropyLoss, SGD optimizer, resnet50, remove fc when tes   ting
   top1: 67%          top5: 92%          top10: 97%

2. triplet_amin.py:
   triplet loss, use TripletMarginLoss, random triplets, Adam optimizer, resnet50 remove fc for both train and test
   top1: 67%          top5: 95%          top10: 99%

3. variant_triplet_amin.py:
   triplet loss, use batch hard triplet loss(2 cameras, each has 64 persons, 3 images in one mini-batch),  Adam optimizer, resnet50 remove fc for both train and test
   top1: 83.75%          top5: 98.75%          top10: 100%

4. resnet_new.py:
   triplet loss, use batch hard triplet loss(2 cameras, each has 64 persons, 3 images in one mini-batch),  Adam optimizer, resnet50 remove last layer, add 2 fc layers for both train and test
   top1: 83.75%          top5: 95%          top10: 98.75%

5. full_variant_triplet_amin.py:
   triplet loss, use batch hard triplet loss(2 cameras, each has 32 persons, 6 images in one mini-batch),  Adam optimizer, resnet50 remove fc for both train and test
   top1: 85%          top5: 95%          top10: 100%

