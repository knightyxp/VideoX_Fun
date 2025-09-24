ps -ef | grep train_lora.py | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep train_1.3b.sh | grep -v grep | awk '{print $2}' | xargs kill -9