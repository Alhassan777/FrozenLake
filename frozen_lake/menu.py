"""
Menu system for the Frozen Lake game
"""
import os
import pygame
import sys
from pygame.locals import *

class MenuSystem:
    """
    Menu system for the Frozen Lake game with integrated auto-solver
    """
    
    def __init__(self, window_size=(800, 600)):
        """
        Initialize the menu system
        
        Args:
            window_size: Size of the window
        """
        # Initialize pygame
        pygame.init()
        
        # Window setup
        self.window_size = window_size
        self.screen = pygame.display.set_mode(self.window_size, pygame.RESIZABLE)
        pygame.display.set_caption("Frozen Lake - Menu")
        
        # Colors
        self.colors = {
            'background': (240, 248, 255),
            'text': (0, 0, 0),
            'button': (135, 206, 250),
            'button_hover': (100, 149, 237),
            'button_text': (0, 0, 0),
            'input_bg': (255, 255, 255),
            'input_text': (0, 0, 0),
            'input_border': (100, 100, 100),
            'border': (100, 100, 100),
            'title': (0, 0, 128),
            'dropdown': (245, 245, 245),
            'dropdown_hover': (220, 220, 220),
            'dropdown_border': (100, 100, 100)
        }
        
        # Fonts
        self.title_font = pygame.font.SysFont('Arial', 36, bold=True)
        self.heading_font = pygame.font.SysFont('Arial', 24, bold=True)
        self.font = pygame.font.SysFont('Arial', 20)
        self.small_font = pygame.font.SysFont('Arial', 16)
        
        # Buttons and inputs
        self.buttons = []
        self.input_boxes = []
        self.active_input = None
        self.checkboxes = []
        self.dropdowns = []
        self.active_dropdown = None
        
        # Game settings
        self.settings = {
            'map_name': '4x4',
            'is_slippery': True,
            'algorithm': 'q_learning',  # q_learning, sarsa, or dqn
            'sound_dir': 'assets',
            'learning_params': {
                'q_learning': {
                    'learning_rate': 0.1,
                    'discount_factor': 0.99,
                    'exploration_rate': 0.1,
                    'training_episodes': 10000
                },
                'sarsa': {
                    'learning_rate': 0.1,
                    'discount_factor': 0.99,
                    'exploration_rate': 0.1,
                    'training_episodes': 10000
                },
                'dqn': {
                    'learning_rate': 0.001,
                    'discount_factor': 0.99,
                    'exploration_rate': 0.1,
                    'training_episodes': 5000,
                    'batch_size': 32,
                    'update_target_every': 100
                }
            }
        }
        
        # Clock for controlling frame rate
        self.clock = pygame.time.Clock()
        
        # Current menu mode
        self.current_mode = 'main'  # 'main', 'game', 'auto_solver'
    
    def create_button(self, rect, text, action, args=None, hover_text=None):
        """
        Create a button
        
        Args:
            rect: Rectangle for the button (x, y, width, height)
            text: Text to display on the button
            action: Function to call when the button is clicked
            args: Arguments to pass to the action function
            hover_text: Text to display when hovering over the button
        """
        if len(rect) == 4:
            rect = pygame.Rect(rect[0], rect[1], rect[2], rect[3])
        
        button = {
            'rect': rect,
            'text': text,
            'action': action,
            'args': args if args is not None else [],
            'hover': False,
            'hover_text': hover_text
        }
        self.buttons.append(button)
        return button
    
    def create_input_box(self, rect, text, key, value, numeric=False):
        """
        Create an input box
        
        Args:
            rect: Rectangle for the input box (x, y, width, height)
            text: Label for the input box
            key: Key in the settings dictionary
            value: Initial value
            numeric: Whether the input should be numeric
        """
        if len(rect) == 4:
            rect = pygame.Rect(rect[0], rect[1], rect[2], rect[3])
            
        input_box = {
            'rect': rect,
            'text': text,
            'value': str(value),
            'key': key,
            'active': False,
            'numeric': numeric
        }
        self.input_boxes.append(input_box)
        return input_box
    
    def create_checkbox(self, rect, text, key, checked=False):
        """
        Create a checkbox
        
        Args:
            rect: Rectangle for the checkbox (x, y, width, height)
            text: Label for the checkbox
            key: Key in the settings dictionary
            checked: Whether the checkbox is checked initially
        """
        # Create a proper Rect object if rect is a tuple
        if isinstance(rect, tuple):
            # If rect is a position tuple (x, y), create a 20x20 checkbox
            if len(rect) == 2:
                rect = pygame.Rect(rect[0], rect[1], 20, 20)
            # If rect is a rect tuple (x, y, w, h), convert to Rect
            elif len(rect) == 4:
                rect = pygame.Rect(rect[0], rect[1], rect[2], rect[3])
        
        checkbox = {
            'rect': rect,
            'text': text,
            'key': key,
            'checked': checked
        }
        self.checkboxes.append(checkbox)
        return checkbox
    
    def create_dropdown(self, rect, text, key, options, selected_option=None):
        """
        Create a dropdown
        
        Args:
            rect: Rectangle for the dropdown (x, y, width, height)
            text: Label for the dropdown
            key: Key in the settings dictionary
            options: List of options
            selected_option: Initially selected option
        """
        if len(rect) == 4:
            rect = pygame.Rect(rect[0], rect[1], rect[2], rect[3])
            
        # Find the initially selected option
        selected_index = 0
        if selected_option is not None:
            if selected_option in options:
                selected_index = options.index(selected_option)
        
        dropdown = {
            'rect': rect,
            'text': text,
            'key': key,
            'options': options,
            'selected_index': selected_index,
            'active': False,
            'open': False,
            'option_rects': []
        }
        
        # Create the option rectangles
        for i in range(len(options)):
            option_rect = pygame.Rect(rect.x, rect.y + rect.height * (i + 1), rect.width, rect.height)
            dropdown['option_rects'].append(option_rect)
        
        self.dropdowns.append(dropdown)
        return dropdown
    
    def draw_title(self, text, y, color=None):
        """
        Draw a title
        
        Args:
            text: Title text
            y: Y position
            color: Color of the title
        """
        if color is None:
            color = self.colors['title']
        
        text_surf = self.title_font.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.window_size[0] // 2, y))
        self.screen.blit(text_surf, text_rect)
        return text_rect.bottom + 10
    
    def draw_heading(self, text, x, y, color=None):
        """
        Draw a heading
        
        Args:
            text: Heading text
            x: X position
            y: Y position
            color: Color of the heading
        """
        if color is None:
            color = self.colors['text']
        
        text_surf = self.heading_font.render(text, True, color)
        text_rect = text_surf.get_rect(topleft=(x, y))
        self.screen.blit(text_surf, text_rect)
        return text_rect.bottom + 10
    
    def draw_text(self, text, x, y, color=None, font=None, centerx=False):
        """
        Draw text
        
        Args:
            text: Text to draw
            x: X position
            y: Y position
            color: Color of the text
            font: Font to use
            centerx: Whether to center horizontally
        """
        if color is None:
            color = self.colors['text']
        
        if font is None:
            font = self.font
        
        text_surf = font.render(text, True, color)
        if centerx:
            text_rect = text_surf.get_rect(midtop=(x, y))
        else:
            text_rect = text_surf.get_rect(topleft=(x, y))
        
        self.screen.blit(text_surf, text_rect)
        return text_rect.bottom + 5
    
    def draw_buttons(self):
        """Draw all buttons"""
        mouse_pos = pygame.mouse.get_pos()
        
        for button in self.buttons:
            # Check if mouse is hovering over button
            button['hover'] = button['rect'].collidepoint(mouse_pos)
            
            # Draw button
            color = self.colors['button_hover'] if button['hover'] else self.colors['button']
            pygame.draw.rect(self.screen, color, button['rect'], border_radius=5)
            pygame.draw.rect(self.screen, self.colors['border'], button['rect'], 2, border_radius=5)
            
            # Draw button text
            text_surf = self.font.render(button['text'], True, self.colors['button_text'])
            text_rect = text_surf.get_rect(center=button['rect'].center)
            self.screen.blit(text_surf, text_rect)
            
            # Draw hover text if any
            if button['hover'] and button['hover_text']:
                hover_text_surf = self.small_font.render(button['hover_text'], True, self.colors['text'])
                hover_text_rect = hover_text_surf.get_rect(midtop=(button['rect'].centerx, button['rect'].bottom + 5))
                self.screen.blit(hover_text_surf, hover_text_rect)
    
    def draw_input_boxes(self):
        """Draw all input boxes"""
        for box in self.input_boxes:
            # Draw label
            label_surf = self.font.render(box['text'], True, self.colors['text'])
            label_rect = label_surf.get_rect(topleft=(box['rect'].x, box['rect'].y - 30))
            self.screen.blit(label_surf, label_rect)
            
            # Draw box
            pygame.draw.rect(self.screen, self.colors['input_bg'], box['rect'])
            pygame.draw.rect(self.screen, self.colors['input_border'], box['rect'], 2)
            
            # Draw text
            text_surf = self.font.render(box['value'], True, self.colors['input_text'])
            text_rect = text_surf.get_rect(midleft=(box['rect'].x + 5, box['rect'].centery))
            
            # Ensure text doesn't extend beyond the input box
            text_rect.width = min(text_rect.width, box['rect'].width - 10)
            self.screen.blit(text_surf, text_rect)
            
            # Draw cursor if active
            if box['active']:
                cursor_pos = box['rect'].x + 5 + text_rect.width
                pygame.draw.line(self.screen, self.colors['input_text'],
                               (cursor_pos, box['rect'].y + 5),
                               (cursor_pos, box['rect'].bottom - 5), 2)
    
    def draw_checkboxes(self):
        """Draw all checkboxes"""
        for checkbox in self.checkboxes:
            # Draw box
            pygame.draw.rect(self.screen, self.colors['input_bg'], checkbox['rect'])
            pygame.draw.rect(self.screen, self.colors['input_border'], checkbox['rect'], 2)
            
            # Draw check if checked
            if checkbox['checked']:
                inner_rect = pygame.Rect(checkbox['rect'].x + 4, checkbox['rect'].y + 4,
                                        checkbox['rect'].width - 8, checkbox['rect'].height - 8)
                pygame.draw.rect(self.screen, self.colors['button'], inner_rect)
            
            # Draw label
            label_surf = self.font.render(checkbox['text'], True, self.colors['text'])
            label_rect = label_surf.get_rect(midleft=(checkbox['rect'].right + 10, checkbox['rect'].centery))
            self.screen.blit(label_surf, label_rect)
    
    def draw_dropdowns(self):
        """Draw all dropdowns"""
        for dropdown in self.dropdowns:
            # Draw the dropdown label
            label_surface = self.font.render(dropdown['text'], True, self.colors['text'])
            label_rect = label_surface.get_rect(topleft=(dropdown['rect'].x, dropdown['rect'].y - 25))
            self.screen.blit(label_surface, label_rect)
            
            # Draw the dropdown box
            pygame.draw.rect(self.screen, self.colors['dropdown'], dropdown['rect'])
            pygame.draw.rect(self.screen, self.colors['dropdown_border'], dropdown['rect'], 2)
            
            # Draw the selected option
            selected_text = dropdown['options'][dropdown['selected_index']]
            selected_surface = self.font.render(selected_text, True, self.colors['text'])
            selected_rect = selected_surface.get_rect(midleft=(dropdown['rect'].x + 10, dropdown['rect'].centery))
            self.screen.blit(selected_surface, selected_rect)
            
            # Draw the dropdown arrow
            arrow_width = 10
            arrow_height = 5
            arrow_x = dropdown['rect'].right - 20
            arrow_y = dropdown['rect'].centery
            pygame.draw.polygon(self.screen, self.colors['text'], [
                (arrow_x, arrow_y - arrow_height/2),
                (arrow_x + arrow_width, arrow_y - arrow_height/2),
                (arrow_x + arrow_width/2, arrow_y + arrow_height/2)
            ])
            
            # If the dropdown is open, draw the options
            if dropdown['open']:
                for i, option_rect in enumerate(dropdown['option_rects']):
                    pygame.draw.rect(self.screen, self.colors['dropdown'], option_rect)
                    pygame.draw.rect(self.screen, self.colors['dropdown_border'], option_rect, 1)
                    
                    option_text = dropdown['options'][i]
                    option_surface = self.font.render(option_text, True, self.colors['text'])
                    option_text_rect = option_surface.get_rect(midleft=(option_rect.x + 10, option_rect.centery))
                    self.screen.blit(option_surface, option_text_rect)
    
    def handle_events(self):
        """
        Handle pygame events
        
        Returns:
            False if the menu should be closed, True otherwise
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            
            # Handle mouse events
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Get the mouse position
                pos = pygame.mouse.get_pos()
                
                # Check if any button was clicked
                for button in self.buttons:
                    if button['rect'].collidepoint(pos):
                        # Call the button's action function with its arguments
                        result = button['action'](*button['args'])
                        
                        # If the action function returns False, close the menu
                        if result is False:
                            return False
                        
                # Check if any input box was clicked
                for box in self.input_boxes:
                    # Deactivate all input boxes
                    box['active'] = False
                    
                    # If this box was clicked, activate it
                    if box['rect'].collidepoint(pos):
                        box['active'] = True
                        self.active_input = box
                
                # Check if any checkbox was clicked
                for checkbox in self.checkboxes:
                    if checkbox['rect'].collidepoint(pos):
                        checkbox['checked'] = not checkbox['checked']
                        
                        # Update the settings dictionary
                        if '.' in checkbox['key']:
                            # Handle nested keys
                            parts = checkbox['key'].split('.')
                            self.settings[parts[0]][parts[1]] = checkbox['checked']
                        else:
                            self.settings[checkbox['key']] = checkbox['checked']
                
                # Check if any dropdown was clicked
                for dropdown in self.dropdowns:
                    # If the dropdown header was clicked, toggle the open state
                    if dropdown['rect'].collidepoint(pos):
                        dropdown['open'] = not dropdown['open']
                        if dropdown['open']:
                            self.active_dropdown = dropdown
                        else:
                            self.active_dropdown = None
                    # If an option was clicked, select it
                    elif dropdown['open']:
                        for i, option_rect in enumerate(dropdown['option_rects']):
                            if option_rect.collidepoint(pos):
                                dropdown['selected_index'] = i
                                dropdown['open'] = False
                                self.active_dropdown = None
                                
                                # Update the settings dictionary
                                if '.' in dropdown['key']:
                                    # Handle nested keys
                                    parts = dropdown['key'].split('.')
                                    self.settings[parts[0]][parts[1]] = dropdown['options'][i]
                                else:
                                    self.settings[dropdown['key']] = dropdown['options'][i]
                
                # If a click happened outside of any open dropdown, close it
                if self.active_dropdown is not None and not self.active_dropdown['rect'].collidepoint(pos):
                    outside_options = True
                    for option_rect in self.active_dropdown['option_rects']:
                        if option_rect.collidepoint(pos):
                            outside_options = False
                            break
                    
                    if outside_options:
                        self.active_dropdown['open'] = False
                        self.active_dropdown = None
                
                # If we didn't click on any input box, deactivate all of them
                if self.active_input is None:
                    for box in self.input_boxes:
                        box['active'] = False
            
            # Handle keyboard events
            if event.type == pygame.KEYDOWN:
                # If an input box is active, handle key presses
                if self.active_input is not None:
                    if event.key == pygame.K_RETURN:
                        # Deactivate the input box
                        self.active_input['active'] = False
                        self.active_input = None
                    elif event.key == pygame.K_BACKSPACE:
                        # Remove the last character
                        self.active_input['value'] = self.active_input['value'][:-1]
                    else:
                        # Add the character to the input box
                        # If the input is numeric, only allow numbers and decimal points
                        if self.active_input['numeric']:
                            if event.unicode.isdigit() or event.unicode == '.':
                                self.active_input['value'] += event.unicode
                        else:
                            self.active_input['value'] += event.unicode
            
            # Handle window resize
            if event.type == pygame.VIDEORESIZE:
                # Update the window size
                self.window_size = event.size
                self.screen = pygame.display.set_mode(self.window_size, pygame.RESIZABLE)
                
                # Re-setup the menu
                self.setup_menu()
        
        # Update the settings dictionary based on input box values
        self.update_settings_from_inputs()
        
        # Return True to continue running the menu
        return True

    def update_settings_from_inputs(self):
        """Update settings dictionary from input boxes, checkboxes, and dropdowns"""
        # Update from checkboxes
        for checkbox in self.checkboxes:
            if '.' in checkbox['key']:
                # Handle nested keys
                parts = checkbox['key'].split('.')
                self.settings[parts[0]][parts[1]] = checkbox['checked']
            else:
                self.settings[checkbox['key']] = checkbox['checked']
        
        # Update from dropdowns
        for dropdown in self.dropdowns:
            if dropdown['key'] == '_algorithm_display':
                # Handle the special case for algorithm dropdown with display names
                algorithm_display_name = dropdown['options'][dropdown['selected_index']]
                # Convert display name to internal name
                algorithm_internal_name = None
                if algorithm_display_name == 'Q-Learning':
                    algorithm_internal_name = 'q_learning'
                elif algorithm_display_name == 'SARSA':
                    algorithm_internal_name = 'sarsa'
                elif algorithm_display_name == 'DQN':
                    algorithm_internal_name = 'dqn'
                
                if algorithm_internal_name:
                    self.settings['algorithm'] = algorithm_internal_name
            elif '.' in dropdown['key']:
                # Handle nested keys
                parts = dropdown['key'].split('.')
                self.settings[parts[0]][parts[1]] = dropdown['options'][dropdown['selected_index']]
            else:
                self.settings[dropdown['key']] = dropdown['options'][dropdown['selected_index']]
        
        # Update from input boxes
        for box in self.input_boxes:
            if '.' not in box['key']:
                # Convert to the appropriate type
                if box['numeric']:
                    try:
                        # Try as float first, then as int if it's a whole number
                        value = float(box['value'])
                        if value.is_integer():
                            value = int(value)
                        self.settings[box['key']] = value
                    except ValueError:
                        pass
                else:
                    self.settings[box['key']] = box['value']
            else:
                # Handle nested keys (like learning_params.q_learning.learning_rate)
                parts = box['key'].split('.')
                
                # Convert to the appropriate type
                if box['numeric']:
                    try:
                        # Try as float first, then as int if it's a whole number
                        value = float(box['value'])
                        if value.is_integer():
                            value = int(value)
                        
                        if len(parts) == 2:
                            self.settings[parts[0]][parts[1]] = value
                        elif len(parts) == 3:
                            self.settings[parts[0]][parts[1]][parts[2]] = value
                    except ValueError:
                        pass
                else:
                    if len(parts) == 2:
                        self.settings[parts[0]][parts[1]] = box['value']
                    elif len(parts) == 3:
                        self.settings[parts[0]][parts[1]][parts[2]] = box['value']

    def draw(self):
        """Draw the menu"""
        # Fill the screen with the background color
        self.screen.fill(self.colors['background'])
        
        # Draw all components
        self.draw_buttons()
        self.draw_input_boxes()
        self.draw_checkboxes()
        self.draw_dropdowns()
        
        # Update the display
        pygame.display.flip()

    def setup_game_menu(self):
        """Set up the game menu with simplified options for human players"""
        # Clear existing components
        self.buttons = []
        self.input_boxes = []
        self.checkboxes = []
        self.dropdowns = []
        
        # Always disable RL for the regular game mode
        self.settings['enable_rl'] = False
        
        # Title
        y = self.draw_title("Frozen Lake - Play Game", 50)
        
        # Game settings section
        y = self.draw_heading("Game Settings", 50, y + 20)
        
        # Map size options - only include valid maps supported by Gymnasium
        map_sizes = ['4x4', '8x8']
        self.create_dropdown((50, y + 30, 150, 30), "Map Size", "map_name", map_sizes, self.settings['map_name'])
        
        # Slippery ice checkbox
        self.create_checkbox((250, y + 30), "Slippery Ice", "is_slippery", checked=self.settings['is_slippery'])
        
        y += 100
        
        # Start button
        self.create_button((200, y, 200, 50), "Start Game", self._start_game, 
                         hover_text="Start the Frozen Lake game with the current settings")
        
        # Back button
        self.create_button((450, y, 150, 50), "Back", self._back_to_main, 
                         hover_text="Return to the main menu")

    def setup_auto_solver_menu(self):
        """Set up the auto solver menu with algorithm selection and parameters"""
        # Clear existing components
        self.buttons = []
        self.input_boxes = []
        self.checkboxes = []
        self.dropdowns = []
        
        # Always enable RL for the auto-solver mode
        self.settings['enable_rl'] = True
        
        # Title
        y = self.draw_title("Frozen Lake - Auto-Solver", 50)
        
        # Environment settings section
        y = self.draw_heading("Environment Settings", 50, y + 20)
        
        # Map size options - only include valid maps supported by Gymnasium
        map_sizes = ['4x4', '8x8']
        self.create_dropdown((50, y + 30, 150, 30), "Map Size", "map_name", map_sizes, self.settings['map_name'])
        
        # Slippery ice checkbox
        self.create_checkbox((250, y + 30), "Slippery Ice", "is_slippery", checked=self.settings['is_slippery'])
        
        y += 100
        
        # Algorithm settings section
        y = self.draw_heading("Algorithm Settings", 50, y)
        
        # Algorithm selection - map display names to internal names
        algorithms = ['q_learning', 'sarsa', 'dqn']
        algorithm_display_names = ['Q-Learning', 'SARSA', 'DQN']
        
        # Find the current algorithm's index
        current_alg = self.settings['algorithm']
        if current_alg in algorithms:
            selected_index = algorithms.index(current_alg)
        else:
            # If it's already a display name, find that
            if current_alg in algorithm_display_names:
                selected_index = algorithm_display_names.index(current_alg)
            else:
                selected_index = 0  # Default to Q-Learning
        
        # Create dropdown with display names
        alg_dropdown = self.create_dropdown((50, y + 30, 200, 30), "Algorithm", "_algorithm_display", 
                                        algorithm_display_names, algorithm_display_names[selected_index])
        
        y += 100
        
        # Algorithm parameters
        current_alg = self.settings['algorithm']
        if current_alg in algorithm_display_names:
            # Convert display name to internal name
            current_alg = algorithms[algorithm_display_names.index(current_alg)]
        
        y = self.draw_heading(f"Algorithm Parameters", 50, y)
        
        if current_alg == 'q_learning' or current_alg == 'sarsa':
            # Common parameters for Q-Learning and SARSA
            self.create_input_box((50, y + 30, 100, 30), "Learning Rate", f"learning_params.{current_alg}.learning_rate", 
                                self.settings['learning_params'][current_alg]['learning_rate'], numeric=True)
            
            self.create_input_box((300, y + 30, 100, 30), "Discount Factor", f"learning_params.{current_alg}.discount_factor", 
                                self.settings['learning_params'][current_alg]['discount_factor'], numeric=True)
            
            y += 80
            
            self.create_input_box((50, y + 30, 100, 30), "Exploration Rate", f"learning_params.{current_alg}.exploration_rate", 
                                self.settings['learning_params'][current_alg]['exploration_rate'], numeric=True)
            
            self.create_input_box((300, y + 30, 100, 30), "Training Episodes", f"learning_params.{current_alg}.training_episodes", 
                                self.settings['learning_params'][current_alg]['training_episodes'], numeric=True)
        
        elif current_alg == 'dqn':
            # DQN specific parameters
            self.create_input_box((50, y + 30, 100, 30), "Learning Rate", "learning_params.dqn.learning_rate", 
                                self.settings['learning_params']['dqn']['learning_rate'], numeric=True)
            
            self.create_input_box((300, y + 30, 100, 30), "Discount Factor", "learning_params.dqn.discount_factor", 
                                self.settings['learning_params']['dqn']['discount_factor'], numeric=True)
            
            y += 80
            
            self.create_input_box((50, y + 30, 100, 30), "Exploration Rate", "learning_params.dqn.exploration_rate", 
                                self.settings['learning_params']['dqn']['exploration_rate'], numeric=True)
            
            self.create_input_box((300, y + 30, 100, 30), "Training Episodes", "learning_params.dqn.training_episodes", 
                                self.settings['learning_params']['dqn']['training_episodes'], numeric=True)
            
            y += 80
            
            self.create_input_box((50, y + 30, 100, 30), "Batch Size", "learning_params.dqn.batch_size", 
                                self.settings['learning_params']['dqn']['batch_size'], numeric=True)
            
            self.create_input_box((300, y + 30, 100, 30), "Update Target", "learning_params.dqn.update_target_every", 
                                self.settings['learning_params']['dqn']['update_target_every'], numeric=True)
        
        y += 100
        
        # Start button
        self.create_button((200, y, 200, 50), "Start Auto-Solver", self._start_auto_solver, 
                         hover_text="Train the agent and solve the environment")
        
        # Back button
        self.create_button((450, y, 150, 50), "Back", self._back_to_main, 
                         hover_text="Return to the main menu")

    def setup_main_menu(self):
        """Set up the main menu with clear separation between game and auto-solver options"""
        # Clear existing components
        self.buttons = []
        self.input_boxes = []
        self.checkboxes = []
        self.dropdowns = []
        
        # Title
        y = self.draw_title("Frozen Lake", 50)
        y += 20
        
        # Subtitle
        subtitle_text = "Choose a mode to play:"
        subtitle_surf = self.heading_font.render(subtitle_text, True, self.colors['text'])
        subtitle_rect = subtitle_surf.get_rect(center=(self.window_size[0] // 2, y + 20))
        self.screen.blit(subtitle_surf, subtitle_rect)
        
        # Menu options
        button_width = 300
        button_height = 60
        button_x = (self.window_size[0] - button_width) // 2
        
        # Play game button
        y += 80
        self.create_button((button_x, y, button_width, button_height), "Play Game", self._show_game_menu, 
                         hover_text="Play Frozen Lake manually")
        
        # Description for play game
        desc_text = "Play the game yourself with simple controls"
        desc_surf = self.font.render(desc_text, True, self.colors['text'])
        desc_rect = desc_surf.get_rect(center=(self.window_size[0] // 2, y + button_height + 20))
        self.screen.blit(desc_surf, desc_rect)
        
        # Auto-solver button
        y += 120
        self.create_button((button_x, y, button_width, button_height), "Auto-Solver", self._show_auto_solver_menu, 
                         hover_text="Use RL algorithms to automatically solve the environment")
        
        # Description for auto-solver
        desc_text = "Use reinforcement learning to solve the puzzle"
        desc_surf = self.font.render(desc_text, True, self.colors['text'])
        desc_rect = desc_surf.get_rect(center=(self.window_size[0] // 2, y + button_height + 20))
        self.screen.blit(desc_surf, desc_rect)
        
        # Exit button
        y += 120
        self.create_button((button_x, y, button_width, button_height), "Exit", self._exit, 
                         hover_text="Exit the application")

    def _show_game_menu(self):
        """Show the game menu"""
        self.current_mode = 'game'
        self.setup_game_menu()
    
    def _show_auto_solver_menu(self):
        """Show the auto-solver menu"""
        self.current_mode = 'auto_solver'
        self.setup_auto_solver_menu()
    
    def _back_to_main(self):
        """Go back to the main menu"""
        self.current_mode = 'main'
        self.setup_main_menu()
    
    def _exit(self):
        """Exit the application"""
        pygame.quit()
        return False
    
    def _start_game(self):
        """Start the game with current settings"""
        # This will be overridden by the game controller
        print("Starting game with settings:", self.settings)
        return True
    
    def _start_auto_solver(self):
        """Start the auto-solver with current settings"""
        # This will be overridden by the game controller
        print("Starting auto-solver with settings:", self.settings)
        return True

    def setup_menu(self):
        """Set up the menu components"""
        # Set up the main menu by default
        self.setup_main_menu()

    def run(self):
        """
        Run the menu loop
        
        Returns:
            The selected settings when a game is started
        """
        running = True
        while running:
            # Handle events
            running = self.handle_events()
            
            # Draw the menu
            self.draw()
            
            # Limit the frame rate
            self.clock.tick(60)
        
        # Return the settings when the menu is closed
        return self.settings 