ps -ef | grep predict_v2v_json.py | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep test_lora_1.3b.sh | grep -v grep | awk '{print $2}' | xargs kill -9