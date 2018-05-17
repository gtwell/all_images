#!/bin/bash

# https://www.cyberciti.biz/faq/bash-for-loop-array/
attributes=(coat_length_labels
            collar_design_labels
	        lapel_design_labels
	        neck_design_labels
	        neckline_design_labels
	        pant_length_labels
	        skirt_length_labels
	        sleeve_length_labels)

for attr in "${attributes[@]}"
do
	printf "Training %s...\n" $attr
	python bin/transfer_learning.py \
		--attribute $attr \
		--epochs 24 \
		--model inceptionresnetv2 \
		--img_size 299 \
		--save_folder inceptionresnetv2_a \
		--batch_size 24 \
		--csv_folder fashionAI_a
done
exit 0
