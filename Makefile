# 使い方
help:
	@echo パーツセグメンテーションの学習を行う
	@echo " $$ make train"
	@echo 学習済みモデルを用いて評価を行う。
	@echo " $$ make eval"
	@echo "フォーマットする"
	@echo " $$ make format"
	@echo "スタイルチェックを行う"
	@echo " $$ make check_style"

train:
	python3 train.py --batch_size 16 --class_choice airplane

predict:
	python3 predict.py --exp_name=partseg_airplane_eval --class_choice=airplane --model_path=outputs/partseg_airplane/models/model.t7

format:
	autopep8 --in-place --recursive *.py

check_style:
	pycodestyle *.py
	pydocstyle *.py