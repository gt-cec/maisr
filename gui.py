# Currently unused. Might move GUI element creation here later - Ryan
import pygame
import sys

class Button:
    def __init__(self, text, x, y, width, height):
        self.text = text
        self.x, self.y = x,y
        self.width, self.height = width, height
        self.rect = pygame.Rect(x,y,width,height)
        self.color = (255, 120, 80)
        self.default_color = (255, 120, 80)
        self.border_color = (0,0,0)
        self.unlatched_border_color = (40,40,40)
        self.border_width = 3
        self.unlatched_border_width = 2
        self.font = pygame.font.SysFont(None, 32)
        self.is_latched = False

    def draw(self,win):
        if self.is_latched:
            self.color = (self.color[0]*.7, self.color[1]*.7, self.color[2]*.7)
            self.border_width = 4

            pygame.draw.line(win, (0,0,0), (self.x - 3, self.y - 3), (self.x + self.width + 3, self.y - 3),self.border_width)  # Top border
            pygame.draw.line(win, self.border_color, (self.x - 3, self.y + self.height + 1),(self.x + self.width + 3, self.y + self.height + 1), self.border_width)  # Bottom border
            pygame.draw.line(win, self.border_color, (self.x - 3, self.y - 3), (self.x - 3, self.y + self.height + 1),self.border_width)  # Left border
            pygame.draw.line(win, self.border_color, (self.x + self.width + 1, self.y - 3),(self.x + self.width + 1, self.y + self.height + 1), self.border_width)  # Right border

        else: #if self.text == "FULL":
            pygame.draw.line(win, self.unlatched_border_color, (self.x - 3+1, self.y - 3+1), (self.x + self.width + 0, self.y - 3+1),self.unlatched_border_width)  # Top border
            pygame.draw.line(win, self.unlatched_border_color, (self.x - 3+1, self.y + self.height),(self.x + self.width, self.y + self.height), self.unlatched_border_width)  # Bottom border
            pygame.draw.line(win, self.unlatched_border_color, (self.x - 3+1, self.y - 3+1), (self.x - 3+1, self.y + self.height + 1),self.unlatched_border_width)  # Left border
            pygame.draw.line(win, self.unlatched_border_color, (self.x + self.width, self.y - 3+1),(self.x + self.width, self.y + self.height + 1), self.unlatched_border_width)  # Right border

        pygame.draw.rect(win, self.color, self.rect)
        text_surface = self.font.render(self.text, True, (0,0,0))
        text_rect = text_surface.get_rect(center=(self.rect.x + self.rect.width // 2, self.rect.y + self.rect.height // 2))
        win.blit(text_surface, text_rect)

    def is_clicked(self,pos):
        return self.rect.collidepoint(pos)


########################################################################################################################
################################################     BUTTON HANDLER    #################################################
########################################################################################################################


class ButtonHandler:
    def __init__(self, env, agent0_policy, game_logger=None, log_data=False):
        self.env = env
        self.agent0_policy = agent0_policy
        self.game_logger = game_logger
        self.log_data = log_data
        self.gameplan_command_history = []

    def handle_mouse_click(self, mouse_position, time_sec):
        """Handle mouse clicks and button interactions"""
        # Handle game area clicks
        if self._is_in_game_area(mouse_position):
            return self._handle_game_area_click(mouse_position, time_sec)

        # Handle button clicks
        elif self.env.target_id_button.is_clicked(mouse_position):
            return self._handle_target_id_button(mouse_position, time_sec)
        elif self.env.wez_id_button.is_clicked(mouse_position):
            return self._handle_wez_id_button(mouse_position, time_sec)
        elif self.env.hold_button.is_clicked(mouse_position):
            return self._handle_hold_button(mouse_position, time_sec)
        elif self.env.waypoint_button.is_clicked(mouse_position):
            return self._handle_waypoint_button(mouse_position)
        elif self.env.NW_quad_button.is_clicked(mouse_position) and not self.env.full_quad_button.is_clicked(
                mouse_position):
            return self._handle_nw_quad_button(mouse_position, time_sec)
        elif self.env.NE_quad_button.is_clicked(mouse_position) and not self.env.full_quad_button.is_clicked(
                mouse_position):
            return self._handle_ne_quad_button(mouse_position, time_sec)
        elif self.env.SW_quad_button.is_clicked(mouse_position) and not self.env.full_quad_button.is_clicked(
                mouse_position):
            return self._handle_sw_quad_button(mouse_position, time_sec)
        elif self.env.SE_quad_button.is_clicked(mouse_position) and not self.env.full_quad_button.is_clicked(
                mouse_position):
            return self._handle_se_quad_button(mouse_position, time_sec)
        elif self.env.full_quad_button.is_clicked(mouse_position):
            return self._handle_full_quad_button(mouse_position, time_sec)
        elif self.env.manual_priorities_button.is_clicked(mouse_position):
            return self._handle_manual_priorities_button(mouse_position, time_sec)
        elif self.env.autonomous_button.is_clicked(mouse_position):
            return self._handle_autonomous_button(mouse_position, time_sec)

        return None, None  # No action needed

    def _is_in_game_area(self, mouse_position):
        """Check if click is in the main game area"""
        margin = self.env.config['gameboard border margin']
        size = self.env.config['gameboard size']
        return (margin < mouse_position[0] < size - margin and
                margin < mouse_position[1] < size - margin)

    def _handle_game_area_click(self, mouse_position, time_sec):
        """Handle clicks in the main game area"""
        if self.env.agent_waypoint_clicked:
            if self.log_data:
                self.game_logger.log_mouse_event(mouse_position, "waypoint override", self.env.display_time)
                self.gameplan_command_history.append([time_sec, 'waypoint override', mouse_position])

            self.env.comm_text = 'Moving to waypoint'
            self.env.add_comm_message(self.env.comm_text, is_ai=True)

            agent0_action = mouse_position
            self.agent0_policy.waypoint_override = mouse_position

            if self.agent0_policy.hold_commanded:
                self.agent0_policy.hold_commanded = False
                self.env.button_latch_dict['hold'] = False

            self.env.agent_waypoint_clicked = False
            self.env.button_latch_dict['waypoint'] = False

            return agent0_action, None
        else:  # Set human waypoint
            if self.log_data:
                self.game_logger.log_mouse_event(mouse_position, "human waypoint", self.env.display_time)
            return None, mouse_position

    def _handle_target_id_button(self, mouse_position, time_sec):
        """Handle target ID button click"""
        if self.log_data:
            self.game_logger.log_mouse_event(mouse_position, "target id", self.env.display_time)

        self.agent0_policy.search_type_override = 'target'
        self.agent0_policy.waypoint_override = False
        self.gameplan_command_history.append([time_sec, 'target_id'])

        self.env.button_latch_dict['target_id'] = True
        if self.env.button_latch_dict['target_id']:
            self.env.button_latch_dict['wez_id'] = False
            self.env.button_latch_dict['autonomous'] = False
            self.env.button_latch_dict['hold'] = False  # target id and wez id policies are mutually exclusive
            self.env.button_latch_dict['manual_priorities'] = True

        if self.agent0_policy.hold_commanded:
            self.agent0_policy.hold_commanded = False
            self.env.button_latch_dict['hold'] = False

        self.env.comm_text = 'Beginning target ID'
        self.env.add_comm_message(self.env.comm_text)

        return None, None

    def _handle_wez_id_button(self, mouse_position, time_sec):
        """Handle WEZ ID button click"""
        if self.log_data:
            self.game_logger.log_mouse_event(mouse_position, "wez id", self.env.display_time)

        self.agent0_policy.search_type_override = 'wez'
        self.agent0_policy.waypoint_override = False
        self.gameplan_command_history.append([time_sec, 'wez_id'])

        self.env.button_latch_dict['wez_id'] = True
        if self.env.button_latch_dict['wez_id']:
            self.env.button_latch_dict['target_id'] = False
            self.env.button_latch_dict['autonomous'] = False
            self.env.button_latch_dict['hold'] = False  # target id and wez id policies are mutually exclusive
            self.env.button_latch_dict['manual_priorities'] = True

        if self.agent0_policy.hold_commanded:
            self.agent0_policy.hold_commanded = False
            self.env.button_latch_dict['hold'] = False

        self.env.comm_text = 'Beginning target+WEZ ID'
        self.env.add_comm_message(self.env.comm_text)

        return None, None

    def _handle_hold_button(self, mouse_position, time_sec):
        """Handle hold button click"""
        if not self.agent0_policy.hold_commanded:
            if self.log_data:
                self.game_logger.log_mouse_event(mouse_position, "hold", self.env.display_time)

            self.agent0_policy.hold_commanded = True
            self.agent0_policy.waypoint_override = False
            self.gameplan_command_history.append([time_sec, 'hold'])

            self.env.button_latch_dict['hold'] = True
            self.env.button_latch_dict['autonomous'] = False
            self.env.comm_text = 'Holding'
        else:
            self.agent0_policy.hold_commanded = False
            self.env.button_latch_dict['hold'] = False
            self.env.comm_text = 'Resuming search'
            self.env.button_latch_dict['autonomous'] = True

        self.env.add_comm_message(self.env.comm_text, is_ai=True)
        return None, None

    def _handle_waypoint_button(self, mouse_position):
        """Handle waypoint button click"""
        self.env.button_latch_dict['waypoint'] = True
        self.env.agent_waypoint_clicked = True
        return None, None

    def _handle_nw_quad_button(self, mouse_position, time_sec):
        """Handle NW quadrant button click"""
        if self.log_data:
            self.game_logger.log_mouse_event(mouse_position, "quadrant - NW", self.env.display_time)

        self.agent0_policy.search_quadrant_override = 'NW'
        self.agent0_policy.waypoint_override = False
        self.gameplan_command_history.append([time_sec, 'NW'])

        self.env.button_latch_dict['NW'] = True
        if self.env.button_latch_dict['NW']:
            self.env.button_latch_dict['NE'] = False
            self.env.button_latch_dict['SE'] = False
            self.env.button_latch_dict['SW'] = False
            self.env.button_latch_dict['full'] = False
            self.env.button_latch_dict['autonomous'] = False
            self.env.button_latch_dict['hold'] = False
            self.env.button_latch_dict['manual_priorities'] = True

        if self.agent0_policy.hold_commanded:
            self.agent0_policy.hold_commanded = False
            self.env.button_latch_dict['hold'] = False

        self.env.comm_text = 'Prioritizing NW quadrant'
        self.env.add_comm_message(self.env.comm_text, is_ai=True)
        return None, None

    def _handle_ne_quad_button(self, mouse_position, time_sec):
        """Handle NE quadrant button click"""
        if self.log_data:
            self.game_logger.log_mouse_event(mouse_position, "quadrant - NE", self.env.display_time)

        self.agent0_policy.search_quadrant_override = 'NE'
        self.agent0_policy.waypoint_override = False
        self.gameplan_command_history.append([time_sec, 'NE'])

        self.env.button_latch_dict['NE'] = True
        if self.env.button_latch_dict['NE']:
            self.env.button_latch_dict['NW'] = False
            self.env.button_latch_dict['SE'] = False
            self.env.button_latch_dict['SW'] = False
            self.env.button_latch_dict['full'] = False
            self.env.button_latch_dict['autonomous'] = False
            self.env.button_latch_dict['hold'] = False
            self.env.button_latch_dict['manual_priorities'] = True

        if self.agent0_policy.hold_commanded:
            self.agent0_policy.hold_commanded = False
            self.env.button_latch_dict['hold'] = False

        self.env.comm_text = 'Prioritizing NE quadrant'
        self.env.add_comm_message(self.env.comm_text, is_ai=True)
        return None, None

    def _handle_sw_quad_button(self, mouse_position, time_sec):
        """Handle SW quadrant button click"""
        if self.log_data:
            self.game_logger.log_mouse_event(mouse_position, "quadrant - SW", self.env.display_time)

        self.agent0_policy.search_quadrant_override = 'SW'
        self.agent0_policy.waypoint_override = False
        self.gameplan_command_history.append([time_sec, 'SW'])

        self.env.button_latch_dict['SW'] = True
        if self.env.button_latch_dict['SW']:
            self.env.button_latch_dict['NE'] = False
            self.env.button_latch_dict['SE'] = False
            self.env.button_latch_dict['NW'] = False
            self.env.button_latch_dict['full'] = False
            self.env.button_latch_dict['autonomous'] = False
            self.env.button_latch_dict['hold'] = False
            self.env.button_latch_dict['manual_priorities'] = True

        if self.agent0_policy.hold_commanded:
            self.agent0_policy.hold_commanded = False
            self.env.button_latch_dict['hold'] = False

        self.env.comm_text = 'Prioritizing SW quadrant'
        self.env.add_comm_message(self.env.comm_text, is_ai=True)
        return None, None

    def _handle_se_quad_button(self, mouse_position, time_sec):
        """Handle SE quadrant button click"""
        if self.log_data:
            self.game_logger.log_mouse_event(mouse_position, "quadrant - SE", self.env.display_time)

        self.agent0_policy.search_quadrant_override = 'SE'
        self.agent0_policy.waypoint_override = False
        self.gameplan_command_history.append([time_sec, 'SE'])

        self.env.button_latch_dict['SE'] = True
        if self.env.button_latch_dict['SE']:
            self.env.button_latch_dict['NE'] = False
            self.env.button_latch_dict['SW'] = False
            self.env.button_latch_dict['NW'] = False
            self.env.button_latch_dict['full'] = False
            self.env.button_latch_dict['autonomous'] = False
            self.env.button_latch_dict['hold'] = False
            self.env.button_latch_dict['manual_priorities'] = True

        if self.agent0_policy.hold_commanded:
            self.agent0_policy.hold_commanded = False
            self.env.button_latch_dict['hold'] = False

        self.env.comm_text = 'Prioritizing SE quadrant'
        self.env.add_comm_message(self.env.comm_text, is_ai=True)
        return None, None

    def _handle_full_quad_button(self, mouse_position, time_sec):
        """Handle full quadrant button click"""
        if self.log_data:
            self.game_logger.log_mouse_event(mouse_position, "quadrant - full", self.env.display_time)

        self.agent0_policy.search_quadrant_override = 'full'
        self.agent0_policy.waypoint_override = False
        self.gameplan_command_history.append([time_sec, 'full'])

        self.env.button_latch_dict['full'] = True
        if self.env.button_latch_dict['full']:
            self.env.button_latch_dict['NE'] = False
            self.env.button_latch_dict['SW'] = False
            self.env.button_latch_dict['NW'] = False
            self.env.button_latch_dict['SE'] = False
            self.env.button_latch_dict['autonomous'] = False
            self.env.button_latch_dict['hold'] = False
            self.env.button_latch_dict['manual_priorities'] = True

        if self.agent0_policy.hold_commanded:
            self.agent0_policy.hold_commanded = False
            self.env.button_latch_dict['hold'] = False

        self.env.comm_text = 'Prioritizing full map'
        self.env.add_comm_message(self.env.comm_text, is_ai=True)
        return None, None

    def _handle_manual_priorities_button(self, mouse_position, time_sec):
        """Handle manual priorities button click"""
        if self.log_data:
            self.game_logger.log_mouse_event(mouse_position, "manual priorities", self.env.display_time)

        self.agent0_policy.search_quadrant_override = 'none'
        self.agent0_policy.waypoint_override = False
        self.gameplan_command_history.append([time_sec, 'manual_priorities'])

        self.agent0_policy.search_type_override = 'none'
        self.env.button_latch_dict['manual_priorities'] = True

        if self.env.button_latch_dict['manual_priorities']:
            self.env.button_latch_dict['autonomous'] = False

        if self.agent0_policy.hold_commanded:
            self.agent0_policy.hold_commanded = False
            self.env.button_latch_dict['hold'] = False

        return None, None

    def _handle_autonomous_button(self, mouse_position, time_sec):
        """Handle autonomous button click"""
        if self.log_data:
            self.game_logger.log_mouse_event(mouse_position, "autonomous", self.env.display_time)

        self.agent0_policy.search_quadrant_override = 'none'
        self.agent0_policy.search_type_override = 'none'
        self.agent0_policy.waypoint_override = False
        self.gameplan_command_history.append([time_sec, 'autonomous'])

        self.env.button_latch_dict['autonomous'] = True
        if self.env.button_latch_dict['autonomous']:
            self.env.button_latch_dict['NE'] = False
            self.env.button_latch_dict['SW'] = False
            self.env.button_latch_dict['NW'] = False
            self.env.button_latch_dict['SE'] = False
            self.env.button_latch_dict['full'] = False
            self.env.button_latch_dict['hold'] = False
            self.env.button_latch_dict['target_id'] = False
            self.env.button_latch_dict['wez_id'] = False
            self.env.button_latch_dict['manual_priorities'] = False

        if self.agent0_policy.hold_commanded:
            self.agent0_policy.hold_commanded = False
            self.env.button_latch_dict['hold'] = False

        self.env.comm_text = 'Beginning autonomous search'
        self.env.add_comm_message(self.env.comm_text, is_ai=True)
        return None, None

    def get_command_history(self):
        """Return the command history for logging"""
        return self.gameplan_command_history

########################################################################################################################
###################################################   SCORE WINDOW   ###################################################
########################################################################################################################

class ScoreWindow:
    def __init__(self, score, x, y):
        self.rect = pygame.Rect(x,y,150,70)
        self.color = (200,200,200)
        self.font = pygame.font.SysFont(None, 36)
        self.score = 0

    def draw(self,win):
        black = (0,0,0)
        #pygame.draw.rect(win, self.color, self.rect)
        #text_surface = self.font.render('SCORE', True, black)
        #text_rect = text_surface.get_rect(center=(self.rect.x + self.rect.width // 2, self.rect.y + 0.5*self.rect.height // 2))
        #win.blit(text_surface, text_rect)

        score_text_surface = self.font.render(str(self.score), True, black)
        #score_text_rect = score_text_surface.get_rect(center=(self.rect.x + self.rect.width // 2, self.rect.y + 1.4*self.rect.height // 2))
        score_text_rect = score_text_surface.get_rect(center=(1020 + 700 // 2, 590+240 // 2))
        win.blit(score_text_surface, score_text_rect)

    def update(self,score):
        self.score = score # Note: The truth source for score is env.score. This gets updated from that.

# class HealthWindow:
#     def __init__(self, agent_id, x, y, text, title_color):
#         self.rect = pygame.Rect(x,y,150,70)
#         self.color = (200,200,200)
#         self.font = pygame.font.SysFont(None, 42)
#         self.agent_id = agent_id
#         self.damage = 0
#         self.text = text
#         self.damage_text_color = (0,0,0)
#         self.title_color = title_color
#
#     def draw(self,win):
#         pygame.draw.rect(win, self.color, self.rect)
#         text_surface = self.font.render(self.text, True, self.title_color)
#         text_rect = text_surface.get_rect(center=(self.rect.x + self.rect.width // 2, self.rect.y + 0.5*self.rect.height // 2))
#         win.blit(text_surface, text_rect)
#
#         if self.damage >= 70: self.damage_text_color = (255,0,0)
#         elif self.damage >= 40: self.damage_text_color = (210,160,0)
#
#         health_num_text_surface = self.font.render(str(round(self.damage,1)) + '/10', True,self.damage_text_color)  # TODO: Update with agent health
#         health_num_text_rect = health_num_text_surface.get_rect(center=(self.rect.x + self.rect.width // 2, self.rect.y + 1.4*self.rect.height // 2))
#         win.blit(health_num_text_surface, health_num_text_rect)
#
#     def update(self,damage):
#         self.damage = damage # Note: The truth source for score is env.score. This gets updated from that.


class HealthWindow:
    def __init__(self, agent_id, x, y, text, title_color):
        self.rect = pygame.Rect(x, y, 150, 70)
        self.color = (200, 200, 200)
        self.font = pygame.font.SysFont(None, 42)
        self.agent_id = agent_id
        self.health_points = 10  # Max health
        self.text = text
        self.title_color = title_color

        # Health bar specific properties
        self.bar_width = 130
        self.bar_height = 25
        self.bar_x = x + 10
        self.bar_y = y + 35
        self.segment_width = self.bar_width / 10  # Width of each health segment

    def get_health_color(self, health):
        """Return a color ranging from green to red based on health percentage"""
        if health >= 7:
            return (0, 255, 0)  # Green
        elif health >= 4:
            return (255, 165, 0)  # Orange
        else:
            return (255, 0, 0)  # Red

    def draw(self, win):
        # Draw the background box
        pygame.draw.rect(win, self.color, self.rect)

        # Draw the title
        text_surface = self.font.render(self.text, True, self.title_color)
        text_rect = text_surface.get_rect(center=(self.rect.x + self.rect.width // 2, self.rect.y + 20))
        win.blit(text_surface, text_rect)

        # Draw the empty health bar background
        pygame.draw.rect(win, (100, 100, 100), (self.bar_x, self.bar_y, self.bar_width, self.bar_height))

        # Draw the filled portion of the health bar
        if self.health_points > 0:
            health_width = (self.health_points / 10) * self.bar_width
            health_color = self.get_health_color(self.health_points)
            pygame.draw.rect(win, health_color, (self.bar_x, self.bar_y, health_width, self.bar_height))

        # Draw segment lines
        for i in range(1, 10):
            line_x = self.bar_x + (i * self.segment_width)
            pygame.draw.line(win, (0, 0, 0),
                             (line_x, self.bar_y),
                             (line_x, self.bar_y + self.bar_height),
                             2)

        # Draw border around the health bar
        pygame.draw.rect(win, (0, 0, 0),
                         (self.bar_x, self.bar_y, self.bar_width, self.bar_height),
                         2)

    def update(self, health_points):
        """Update the health value (expects value from 0-10)"""
        self.health_points = health_points

class TimeWindow:
    def __init__(self, x, y, current_time=0, time_limit=240):  # Takes current game time in raw form (milliseconds)
        self.rect = pygame.Rect(x, y, 150, 65)
        self.color = (200, 200, 200)
        self.time_font = pygame.font.SysFont(None, 88)
        self.title_font = pygame.font.SysFont(None, 40)
        self.current_time = current_time
        self.time_limit = time_limit

    def format_time(self, seconds):
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes}:{seconds:02d}"  # :02d ensures seconds are padded with leading zero

    def draw(self, win):
        black = (0, 0, 0)
        pygame.draw.rect(win, self.color, self.rect)

        #timer_title_surface = self.title_font.render('TIME LEFT', True, black)
        #win.blit(timer_title_surface, timer_title_surface.get_rect(center=(self.rect.x + self.rect.width // 2, self.rect.y + 0.5 * self.rect.height // 2)))

        time_remaining = self.time_limit - self.current_time/1000
        time_text = self.format_time(time_remaining)
        time_text_surface = self.time_font.render(time_text, True, black)
        time_text_rect = time_text_surface.get_rect(center=(self.rect.x + self.rect.width // 2, self.rect.y + 1.4*self.rect.height // 2 - 10))
        win.blit(time_text_surface, time_text_rect)

    def update(self,time):
        self.current_time = time


class AgentInfoDisplay:
    def __init__(self, x, y, width=445, height=250):
        self.rect = pygame.Rect(x, y, width, height)
        self.title_rect = pygame.Rect(x, y, width, 40)
        self.content_rect = pygame.Rect(x, y + 40, width, height - 40)
        self.color = (230, 230, 230)
        self.title_color = (200, 200, 200)
        self.title_font = pygame.font.SysFont(None, 36)
        self.content_font = pygame.font.SysFont(None, 24)
        self.text = []
        self.line_height = 30
        self.text_margin = 10

        # Risk level colors
        self.risk_colors = {
            'LOW': ((50, 255, 50), (0, 0, 0)),  # (Green bg, black text)
            'MEDIUM': ((255, 165, 0), (0, 0, 0)),  # (Orange bg, black text)
            'HIGH': ((255, 50, 50), (0, 0, 0)),  # (Red bg, black text)
            'EXTREME': ((0, 0, 0), (255, 255, 255))  # (Black bg, white text)
        }

    def draw(self, window):
        # Draw background rectangles
        pygame.draw.rect(window, self.title_color, self.title_rect)
        pygame.draw.rect(window, self.color, self.content_rect)

        # Draw title
        title_surface = self.title_font.render('AGENT STATUS', True, (0, 0, 0))
        title_rect = title_surface.get_rect(center=(self.title_rect.centerx, self.title_rect.centery))
        window.blit(title_surface, title_rect)

        # Draw content
        y_offset = self.content_rect.y + self.text_margin
        for line in self.text:
            if "RISK LEVEL:" in line:
                # Split the line into label and value
                label, risk_level = line.split(": ")
                risk_level = risk_level.strip()

                # Render the label part
                label_surface = self.content_font.render(label + ": ", True, (0, 0, 0))
                window.blit(label_surface, (self.content_rect.x + self.text_margin, y_offset))

                if risk_level in self.risk_colors:
                    bg_color, text_color = self.risk_colors[risk_level]

                    # Render the risk level to get its width
                    risk_surface = self.content_font.render(risk_level, True, text_color)
                    risk_width = risk_surface.get_width()

                    # Draw background rectangle just for the risk level text
                    risk_rect = pygame.Rect(
                        self.content_rect.x + self.text_margin + label_surface.get_width(),
                        y_offset-3,
                        risk_width+6,
                        self.line_height - 10
                    )
                    pygame.draw.rect(window, bg_color, risk_rect)

                    # Draw the risk level text
                    window.blit(risk_surface, (risk_rect.x, y_offset))
                else:
                    # Fallback if risk level not recognized
                    risk_surface = self.content_font.render(risk_level, True, (0, 0, 0))
                    window.blit(risk_surface,
                                (self.content_rect.x + self.text_margin + label_surface.get_width(), y_offset))
            else:
                # Draw regular lines
                text_surface = self.content_font.render(line, True, (0, 0, 0))
                window.blit(text_surface, (self.content_rect.x + self.text_margin, y_offset))

            y_offset += self.line_height

        # Draw borders
        border_width = 4
        pygame.draw.line(window, (0, 0, 0), (self.rect.x, self.rect.y),
                         (self.rect.x + self.rect.width, self.rect.y), border_width)
        pygame.draw.line(window, (0, 0, 0), (self.rect.x, self.rect.y),
                         (self.rect.x, self.rect.y + self.rect.height), border_width)
        pygame.draw.line(window, (0, 0, 0), (self.rect.x + self.rect.width, self.rect.y),
                         (self.rect.x + self.rect.width, self.rect.y + self.rect.height), border_width)
        pygame.draw.line(window, (0, 0, 0), (self.rect.x, self.rect.y + self.rect.height),
                         (self.rect.x + self.rect.width, self.rect.y + self.rect.height), border_width)

    def update_text(self, new_text):
        self.text = new_text
