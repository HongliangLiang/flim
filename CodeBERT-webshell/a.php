for i in index:
          index = i.numpy().tolist()
      for i in index:
          print(outputs[i])
          print(data['filename'][i])

      # print(outputs)
      # print(pred_choice)
      # print(ids.size())
      time_end = time.time()
      print('totally cost', time_end - time_start)