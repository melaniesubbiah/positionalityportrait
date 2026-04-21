#!/bin/bash

echo "Scoring semantics..."
python scripts/score_semantics.py --input_dir summaries --model $1

echo "Scoring emotions..."
python scripts/score_emotions.py --input_dir summaries --model $1

echo "Scoring themes..."
python scripts/score_themes.py --input_dir summaries --model $1

#echo "Generating full portrait for model: $1"
#python scripts/positionality_portrait.py --model_path $1

echo "Generating subset portrait for model: $1"
python scripts/positionality_portrait.py --model_path $1 --attribute_subset
