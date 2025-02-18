# my_dict_alpha = [{"id": 1, "metrics": {"immersiveness_score": 1, "total_cost": 1}},
#                  {"id": 2, "metrics": {"immersiveness_score": 2, "total_cost": 2}}]

# for i in my_dict_alpha:
#    print(i["id"])
#    print(i["metrics"])
# import collections
# myorderedDict = collections.OrderedDict({"a": 1, "b": 2, "c": 3})
# print(myorderedDict.keys())
# print(myorderedDict.values())
from collections import OrderedDict
import os
import csv
k="./"
row_data = OrderedDict({"a": 1, "b": 2, "c": 3})
def write_to_file(row_data: OrderedDict,path):
         """
         Writes a row of data to the summary CSV file.
         @param row_data: OrderedDict where keys are column names and values are the data to write
         """
         if not os.path.exists(os.path.join(path, "summary.csv")):
            header = row_data.keys()
            with open(os.path.join(path, "summary.csv"), "w") as f:
                  writer = csv.writer(f)
                  writer.writerow(header)
         with open(os.path.join(path, "summary.csv"), "a") as f:
            writer = csv.writer(f)
            writer.writerow(row_data.values())

write_to_file(row_data,k)