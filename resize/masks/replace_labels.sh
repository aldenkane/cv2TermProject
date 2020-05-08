#!/bin/bash
#$ -M xzhong3@nd.edu
#$ -m ae
#$ -r n
#$ -q debug
#$ -N replace_labels    # Specify job name

for name in Tape Sunglasses Spray_Sunscreen Rub_Sunscreen Pillbottle Paintbrush Marker Fuzzy Frisbee Dice_Container 
do
	for num in 1 2 3 4 5 6 7 8 9 10 11
	do
		sed -i 's/sunglasses/Sunglasses/g' c_$name""$num".json"
		sed -i 's/tape/Tape/g' c_$name""$num".json"
		sed -i 's/spray_sunscreen/Spray_Sunscreen/g' c_$name""$num".json"
		sed -i 's/rub_sunscreen/Rub_Sunscreen/g' c_$name""$num".json"
		sed -i 's/pillbottle/Pillbottle/g' c_$name""$num".json"
		sed -i 's/paintbrush/Paintbrush/g' c_$name""$num".json"
		sed -i 's/marker/Marker/g' c_$name""$num".json"
		sed -i 's/fuzzy/Fuzzy/g' c_$name""$num".json"
		sed -i 's/frisbee/Frisbee/g' c_$name""$num".json"
		sed -i 's/dice_container/Dice_Container/g' c_$name""$num".json"
		
		sed -i 's/sunglasses/Sunglasses/g' m_$name""$num".json"
		sed -i 's/tape/Tape/g' m_$name""$num".json"
		sed -i 's/spray_sunscreen/Spray_Sunscreen/g' m_$name""$num".json"
		sed -i 's/rub_sunscreen/Rub_Sunscreen/g' m_$name""$num".json"
		sed -i 's/pillbottle/Pillbottle/g' m_$name""$num".json"
		sed -i 's/paintbrush/Paintbrush/g' m_$name""$num".json"
		sed -i 's/marker/Marker/g' m_$name""$num".json"
		sed -i 's/fuzzy/Fuzzy/g' m_$name""$num".json"
		sed -i 's/frisbee/Frisbee/g' m_$name""$num".json"
		sed -i 's/dice_container/Dice_Container/g' m_$name""$num".json"
	done
done
