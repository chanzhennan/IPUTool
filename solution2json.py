import json

def ReadTxtName(rootdir):
    lines = []
    with open(rootdir, 'r') as file_to_read:
        while True:
            line = file_to_read.readline()
            if not line:
                break
            line = line.strip('\n')
            lines.append(line)
    return lines



def gen_json():
  data = ReadTxtName('solution_list.txt')
  manual_sharding = [[], []]
  i = 0
  for d in data:
    i += 1
    _id = int(d.split(":")[1])
    name = d.split(":")[0]
    print(_id, name, i)
    manual_sharding[_id].append(name)

  json_file = {"batch_size": 1,
  "num_ipus": 2,
  "precision_mode":"FP16",
  "output_addition_list":[
      "query_attenqc_cls/dense/Relu:0",
      "query_attenqt_cls/dense/Relu:0"],
  "manual_sharding":'true',
 "manual_sharding_config": {
    "num_shards": 2,
    "shard_patterns":manual_sharding},
  "nodes_blacklist": [
      "TensorDict",
      "Assert/Assert"]}

  with open("./json/manual_sharding_fp16.json","w") as f:
    json.dump(json_file, f)



if __name__ == "__main__":
  gen_json()
