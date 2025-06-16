# Handle new button clicks (add this in your mouse event handling section)
if event.type == pygame.MOUSEBUTTONDOWN:
    mouse_pos = pygame.mouse.get_pos()

    if self.info_button_1.is_clicked(mouse_pos):
        self.button_latch_dict['info_button_1'] = not self.button_latch_dict['info_button_1']
        self.id_requested = True
        print("Info Button 1 clicked")

    elif self.info_button_2.is_clicked(mouse_pos):
        self.button_latch_dict['info_button_2'] = not self.button_latch_dict['info_button_2']
        self.id_requested = True
        print("Info Button 2 clicked")

    elif self.info_button_3.is_clicked(mouse_pos):
        self.button_latch_dict['info_button_3'] = not self.button_latch_dict['info_button_3']
        self.id_requested = True
        print("Info Button 3 clicked")