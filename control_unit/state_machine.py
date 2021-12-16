#
# gui = gui.gui()
# gui.create_window()
# time.sleep(10)
#
# found = init_state.find_human()
# skip_scan = False
# state = []
#
#
#
# while 1:
#
#     if not found:
#         print("Abort no human found")
#         break
#     time.sleep(2)
#     if not skip_scan:
#         state = query_state.scan_human()
#         time.sleep(2)
#
#     song = select_song_state.select_song(state)
#     if not skip_scan:
#         reward = play_song_state.play_song(song)
#         training_state.train(state, reward)
#
#     answer = questioning_state.decideForState()
#
#     if answer == 'down':
#         print('Okay another round')
#         skip_scan = False
#     elif answer == 'up':
#         print('Okay here comes another song to your mood')
#         skip_scan = True
#     elif answer == 'stop':
#         print('Okay Goodbye! I am happy to see u again!')
#         break