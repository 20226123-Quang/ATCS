class SignalController:
    """Base class for traffic signal control strategies"""
    
    def apply_control(self, tls_id, current_time, **kwargs):
        """
        Apply traffic signal control strategy
        Subclasses must implement this method
        """
        raise NotImplementedError("Subclasses must implement apply_control()")
    
    def reset(self, tls_id, current_time):
        """
        Reset internal state when transitioning between strategies
        Optional - subclasses can override if needed
        """
        pass
