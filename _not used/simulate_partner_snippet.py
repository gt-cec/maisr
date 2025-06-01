def reset():
	 self.partner_agent = AutonomousPolicy()

def step():
	partner_action = self.partner_agent.choose_action(self.observation)
	self.partner_agent.move(partner_action)


def get_observation(self):
	# Add partner