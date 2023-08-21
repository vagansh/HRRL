from hyperparam import Hyperparam
import torch


TensorTorch = type(torch.Tensor())

class HomogeneousZeta:
    """Homogeneous to the state (internal + external) of the agent."""
    def __init__(self, hyperparam: Hyperparam) -> None:
        self.hp = hyperparam
        self.tensor = torch.zeros(self.hp.cst_agent.zeta_shape)

    def get_resource(self, res: int):
        assert res < self.hp.difficulty.n_resources
        return float(self.tensor[self.hp.cst_agent.features_to_index[f"resource_{res}"]])

    def set_resource(self, res: int, val: float):
        assert res < self.hp.difficulty.n_resources
        self.tensor[self.hp.cst_agent.features_to_index[f"resource_{res}"]] = val

    @property
    def muscular_fatigue(self) -> float:
        return float(self.tensor[self.hp.cst_agent.features_to_index["muscular_fatigue"]])

    @muscular_fatigue.setter
    def muscular_fatigue(self, val: float) -> None:
        self.tensor[self.hp.cst_agent.features_to_index["muscular_fatigue"]] = val

    @property
    def sleep_fatigue(self) -> float:
        return float(self.tensor[self.hp.cst_agent.features_to_index["sleep_fatigue"]])

    @sleep_fatigue.setter
    def sleep_fatigue(self, val: float) -> None:
        self.tensor[self.hp.cst_agent.features_to_index["sleep_fatigue"]] = val

    @property
    def x(self) -> float:
        return float(self.tensor[self.hp.cst_agent.features_to_index["x"]])
    
    @x.setter
    def x(self, val: float) -> None:
        self.tensor[self.hp.cst_agent.features_to_index["x"]] = val

    @property
    def y(self) -> float:
        return float(self.tensor[self.hp.cst_agent.features_to_index["y"]])
    
    @y.setter
    def y(self, val: float) -> None:
        self.tensor[self.hp.cst_agent.features_to_index["y"]] = val

    @property
    def homeostatic(self) -> TensorTorch:
        return self.tensor[self.hp.cst_agent.features_to_index["homeostatic"]]

    @homeostatic.setter
    def homeostatic(self, val: TensorTorch) -> None:
        self.tensor[self.hp.cst_agent.features_to_index["homeostatic"]] = val

    @property
    def non_homeostatic(self) -> TensorTorch:
        return self.tensor[self.hp.cst_agent.features_to_index["non_homeostatic"]]

    @non_homeostatic.setter
    def non_homeostatic(self, val: TensorTorch) -> None:
        self.tensor[self.hp.cst_agent.features_to_index["non_homeostatic"]] = val


class Agent:
    def __init__(self, hyperparam: Hyperparam):
        """Initialize the Agent.
        """
        self.hp = hyperparam

        self.zeta = HomogeneousZeta(self.hp)
        # Setting initial position
        self.zeta.x = self.hp.cst_agent.default_pos_x
        self.zeta.y = self.hp.cst_agent.default_pos_y
        self.zeta.muscular_fatigue = self.hp.cst_agent.min_resource
        self.zeta.sleep_fatigue = self.hp.cst_agent.min_resource

        def set_val_homeo(dic_val):
            homo_zeta = HomogeneousZeta(self.hp)
            for res in range(self.hp.difficulty.n_resources):
                homo_zeta.set_resource(res, dic_val[f"resource_{res}"])
            homo_zeta.muscular_fatigue = dic_val["muscular_fatigue"]
            homo_zeta.sleep_fatigue = dic_val["sleep_fatigue"]
            return homo_zeta
        
        self.x_star = set_val_homeo(self.hp.cst_agent.val_x_star)

        self.coeff_eq_diff = set_val_homeo(self.hp.cst_agent.val_coeff_eq_diff)

    def drive(self, zeta: HomogeneousZeta, epsilon: float = 0.001) -> float:
        """
        Return the Agent's drive which is the distance between the agent's 
        state and the homeostatic set point.
        """
        # in the delta, we only count the internal state.
        # The tree last coordinate do not count in the homeostatic set point.
        delta = zeta.homeostatic
        drive_delta = float(torch.sqrt(epsilon + torch.dot(delta, delta)))

        crt = zeta.homeostatic.detach().numpy()[0:self.hp.difficulty.n_resources]
        hms = self.x_star.homeostatic.detach().numpy()[0:self.hp.difficulty.n_resources]
        print(crt, hms)

        delta = torch.tensor(crt) - torch.tensor(hms)

        drive_delta = float(torch.sqrt(epsilon + torch.dot(delta, delta)))


        return drive_delta
        
    def dynamics(self, zeta: HomogeneousZeta, u: HomogeneousZeta) -> HomogeneousZeta:
        """
        Return the Agent's dynamics which is represented by the f function.
        """
        f = HomogeneousZeta(self.hp)
        f.homeostatic = (self.coeff_eq_diff.homeostatic + u.homeostatic) * \
            (zeta.homeostatic + self.x_star.homeostatic)
        f.non_homeostatic = u.non_homeostatic
        return f

    def euler_method(self, zeta: HomogeneousZeta, u: HomogeneousZeta) -> HomogeneousZeta:
        """Euler method for tiny time steps.
        """
        new_zeta = HomogeneousZeta(self.hp)
        delta_zeta = self.hp.cst_algo.time_step * self.dynamics(zeta, u).tensor
        new_zeta.tensor = zeta.tensor + delta_zeta
        return new_zeta

    def integrate_multiple_steps(self,
                                 duration: float,
                                 zeta: HomogeneousZeta,
                                 control: HomogeneousZeta) -> HomogeneousZeta:
        """We integrate rigorously with an exponential over 
        long time period the differential equation.
        This function is usefull in the case of big actions, 
        such as going direclty to one of the resource.
        """
        x = zeta.homeostatic + self.x_star.homeostatic
        #print("integrate multiple steps", x, zeta.homeostatic, self.x_star.homeostatic)
        rate = self.coeff_eq_diff.homeostatic + control.homeostatic
        #print(rate, self.coeff_eq_diff.homeostatic, control.homeostatic)
        new_x = x * torch.exp(rate * duration)

        #print("new_x", new_x)
        new_zeta_homeo = new_x - self.x_star.homeostatic
        #print("new_zeta_homeo", new_zeta_homeo)

        new_zeta = HomogeneousZeta(self.hp)
        new_zeta.homeostatic = new_x
        new_zeta.non_homeostatic = zeta.non_homeostatic.clone()
        return new_zeta

