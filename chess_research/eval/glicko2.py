"""
glicko2
~~~~~~~

The Glicko2 rating system.

:copyright: (c) 2012 by Heungsub Lee
:license: BSD, see LICENSE for more details.
https://github.com/sublee/glicko2
"""

import math
import sys

__version__ = "0.0.dev"

#: The actual score for win
WIN = 1.0
#: The actual score for draw
DRAW = 0.5
#: The actual score for loss
LOSS = 0.0


MU = 1500
PHI = 500
SIGMA = 0.06
TAU = 1.0
EPSILON = 0.000001


class Rating:
    def __init__(self, mu=MU, phi=PHI, sigma=SIGMA):
        self.mu = mu
        self.phi = phi
        self.sigma = sigma

    def __repr__(self):
        c = type(self)
        args = (c.__module__, c.__name__, self.mu, self.phi, self.sigma)
        return "%s.%s(mu=%.3f, phi=%.3f, sigma=%.3f)" % args


class Glicko2:
    def __init__(self, mu=MU, phi=PHI, sigma=SIGMA, tau=TAU, epsilon=EPSILON):
        self.mu = mu
        self.phi = phi
        self.sigma = sigma
        self.tau = tau
        self.epsilon = epsilon

    def create_rating(self, mu=None, phi=None, sigma=None):
        if mu is None:
            mu = self.mu
        if phi is None:
            phi = self.phi
        if sigma is None:
            sigma = self.sigma
        return Rating(mu, phi, sigma)

    def scale_down(self, rating, ratio=173.7178):
        mu = (rating.mu - self.mu) / ratio
        phi = rating.phi / ratio
        return self.create_rating(mu, phi, rating.sigma)

    def scale_up(self, rating, ratio=173.7178):
        mu = rating.mu * ratio + self.mu
        phi = rating.phi * ratio
        return self.create_rating(mu, phi, rating.sigma)

    def reduce_impact(self, rating):
        """The original form is `g(RD)`. This function reduces the impact of
        games as a function of an opponent's RD.
        """
        return 1.0 / math.sqrt(1 + (3 * rating.phi**2) / (math.pi**2))

    def expect_score(self, rating, other_rating, impact):
        return 1.0 / (1 + math.exp(-impact * (rating.mu - other_rating.mu)))

    def determine_sigma(self, rating, difference, variance):
        """Determines new sigma."""
        phi = rating.phi
        difference_squared = difference**2
        # 1. Let a = ln(s^2), and define f(x)
        alpha = math.log(rating.sigma**2)

        def f(x):
            """This function is twice the conditional log-posterior density of
            phi, and is the optimality criterion.
            """
            tmp = phi**2 + variance + math.exp(x)
            a = math.exp(x) * (difference_squared - tmp) / (2 * tmp**2)
            b = (x - alpha) / (self.tau**2)
            return a - b

        # 2. Set the initial values of the iterative algorithm.
        a = alpha
        if difference_squared > phi**2 + variance:
            b = math.log(difference_squared - phi**2 - variance)
        else:
            k = 1
            while f(alpha - k * math.sqrt(self.tau**2)) < 0:
                k += 1
            b = alpha - k * math.sqrt(self.tau**2)
        # 3. Let fA = f(A) and f(B) = f(B)
        f_a, f_b = f(a), f(b)
        # 4. While |B-A| > e, carry out the following steps.
        # (a) Let C = A + (A - B)fA / (fB-fA), and let fC = f(C).
        # (b) If fCfB < 0, then set A <- B and fA <- fB; otherwise, just set
        #     fA <- fA/2.
        # (c) Set B <- C and fB <- fC.
        # (d) Stop if |B-A| <= e. Repeat the above three steps otherwise.
        while abs(b - a) > self.epsilon:
            c = a + (a - b) * f_a / (f_b - f_a)
            f_c = f(c)
            if f_c * f_b < 0:
                a, f_a = b, f_b
            else:
                f_a /= 2
            b, f_b = c, f_c
        # 5. Once |B-A| <= e, set s' <- e^(A/2)
        return math.exp(1) ** (a / 2)

    def rate(self, rating, series):
        # Step 2. For each player, convert the rating and RD's onto the
        #         Glicko-2 scale.
        rating = self.scale_down(rating)
        # Step 3. Compute the quantity v. This is the estimated variance of the
        #         team's/player's rating based only on game outcomes.
        # Step 4. Compute the quantity difference, the estimated improvement in
        #         rating by comparing the pre-period rating to the performance
        #         rating based only on game outcomes.
        variance_inv = 0
        difference = 0
        if not series:
            # If the team didn't play in the series, do only Step 6
            phi_star = math.sqrt(rating.phi**2 + rating.sigma**2)
            return self.scale_up(self.create_rating(rating.mu, phi_star, rating.sigma))
        for actual_score, other_rating in series:
            other_rating = self.scale_down(other_rating)
            impact = self.reduce_impact(other_rating)
            expected_score = self.expect_score(rating, other_rating, impact)
            variance_inv += impact**2 * expected_score * (1 - expected_score)
            difference += impact * (actual_score - expected_score)
        difference /= variance_inv
        variance = 1.0 / variance_inv
        # Step 5. Determine the new value, Sigma', ot the sigma. This
        #         computation requires iteration.
        sigma = self.determine_sigma(rating, difference, variance)
        # Step 6. Update the rating deviation to the new pre-rating period
        #         value, Phi*.
        phi_star = math.sqrt(rating.phi**2 + sigma**2)
        # Step 7. Update the rating and RD to the new values, Mu' and Phi'.
        phi = 1.0 / math.sqrt(1 / phi_star**2 + 1 / variance)
        mu = rating.mu + phi**2 * (difference / variance)
        # Step 8. Convert ratings and RD's back to original scale.
        return self.scale_up(self.create_rating(mu, phi, sigma))

    def rate_1vs1(self, rating1, rating2, drawn=False):
        return (
            self.rate(rating1, [(DRAW if drawn else WIN, rating2)]),
            self.rate(rating2, [(DRAW if drawn else LOSS, rating1)]),
        )

    def quality_1vs1(self, rating1, rating2):
        expected_score1 = self.expect_score(
            rating1, rating2, self.reduce_impact(rating1)
        )
        expected_score2 = self.expect_score(
            rating2, rating1, self.reduce_impact(rating2)
        )
        expected_score = (expected_score1 + expected_score2) / 2
        return 2 * (0.5 - abs(0.5 - expected_score))


class GlickoCalc:
    def __init__(self, stockfish_level):
        stockfish_elos = {
            "1": 1554,
            "2": 1759,
            "3": 1867,
            "4": 2006,
            "5": 2148,
        }
        self.session_state = {
            "nanogpt_rating": 1500,
            "stockfish_rating": stockfish_elos[str(stockfish_level)],
            "nanogpt_rd": 500,
            "stockfish_rd": 0,
            "nanogpt_v": 0.06,
            "stockfish_v": 0.06,
            "tau": 0.5,
        }

    def rating_update(self, p):
        self.session_state["nanogpt_rating"] = round(p.mu, 8)
        self.session_state["nanogpt_rd"] = round(p.phi, 8)
        self.session_state["nanogpt_v"] = round(p.sigma, 8)

    def glicko2_update(self, result):
        """Update the glicko 2 rating.

        :param result: If result == 0, this means stockfish won. If result == 1, this means nanogpt won. If result == 2, this means draw.
        :type result: int
        """
        result_options = ["#1 wins", "#2 wins", "draw"]
        env = Glicko2(tau=float(self.session_state["tau"]))

        r2 = env.create_rating(
            self.session_state["nanogpt_rating"],
            self.session_state["nanogpt_rd"],
            self.session_state["nanogpt_v"],
        )
        r1 = env.create_rating(
            self.session_state["stockfish_rating"],
            self.session_state["stockfish_rd"],
            self.session_state["stockfish_v"],
        )

        p = [None, None]
        if result_options[int(result)] == "#1 wins":  # stockfish won
            p[0], p[1] = env.rate_1vs1(r1, r2, drawn=False)
        elif result_options[int(result)] == "#2 wins":  # nanogpt won
            p[1], p[0] = env.rate_1vs1(r2, r1, drawn=False)
        else:  # draw
            p[0], p[1] = env.rate_1vs1(r1, r2, drawn=True)

        self.rating_update(p[0])
        return self.current_elo, self.current_deviation

    @property
    def current_elo(self):
        return round(self.session_state["nanogpt_rating"])

    @property
    def current_deviation(self):
        return round(
            self.session_state["nanogpt_rd"]
        )  # note that the lowest this can go to is 60, or 59.73 precisely


if __name__ == "__main__":
    # python glicko2.py result stockfish_level
    g = GlickoCalc(1)  # stockfish level
    print(g.glicko2_update(0))  # nanogpt won
    print(g.glicko2_update(1))  # nanogpt loss
    print(g.glicko2_update(2))  # draw
