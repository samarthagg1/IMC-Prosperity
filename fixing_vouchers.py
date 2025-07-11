import json
import numpy as np
from collections import deque
from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation, Symbol,Listing,Observation,Trade,ProsperityEncoder
from typing import Any, Dict, TypeAlias

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

class Product:
    GIFT_BASKET = "PICNIC_BASKET1"
    CHOCOLATE = "PICNIC_BASKET2"
    STRAWBERRIES = "CROISSANTS"
    ROSES = "JAMS"
    DJEMBES = "DJEMBES"
    SYNTHETIC = "SYNTHETIC"
    SPREAD = "SPREAD"
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS" 

BASKET_WEIGHTS = {
    Product.CHOCOLATE: 1,
    Product.STRAWBERRIES: 2,
    Product.ROSES: 1,
    Product.DJEMBES: 1,
}

BASKET_PARAMS = {
    Product.SPREAD: {
        "default_spread_mean": -114.291,
        "default_spread_std": 154.89,
        "spread_std_window": 150,
        "zscore_threshold": 7.0,
        "current_std": 80,
        "target_position": 60,
    },
}

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."

logger = Logger()

class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit

    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> tuple[list[Order], int]:
        self.orders = []
        self.conversions = 0

        self.act(state)

        return self.orders, self.conversions

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def convert(self, amount: int) -> None:
        self.conversions += amount

    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass

class MagnificentMacaronsStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.last_s_index = 100
        
        self.params = {
            "reference_price": 687.0,
            "buy_threshold_offset": 5.0,
            "sell_threshold_offset": 5.0,
            "exit_offset": 1.0,
            "position_limit": limit,
            "min_order_size": 5,
            "scaling_factor": 2.5,
            "tier1_threshold": 7.0,
            "tier2_threshold": 14.0,
            "tier3_threshold": 20.0,
            "tier1_size": 75,
            "tier2_size": 75,
            "tier3_size": 75,
            "partial_exit_ratio": 0.5
        }
    
    def _calculate_order_size(self, deviation_amount):
        """Calculates order size based on deviation tiers."""
        if deviation_amount >= self.params["tier3_threshold"]:
            return self.params["tier3_size"]
        elif deviation_amount >= self.params["tier2_threshold"]:
            return self.params["tier2_size"]
        elif deviation_amount >= self.params["tier1_threshold"]:
            return self.params["tier1_size"]
        else:
            base_size = int(self.params["min_order_size"] + (deviation_amount * self.params["scaling_factor"]))
            return max(self.params["min_order_size"], base_size)

    def _apply_mean_reversion(self, state):
        """Applies the dynamic mean reversion strategy."""
        current_position = state.position.get(self.symbol, 0)
        position_limit = self.params["position_limit"]
        order_depth = state.order_depths[self.symbol]

        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        
        if not buy_orders or not sell_orders:
            return
            
        best_bid = buy_orders[0][0]
        best_ask = sell_orders[0][0]

        mid_price = (best_bid + best_ask) / 2
        reference_price = self.params["reference_price"]
        deviation = mid_price - reference_price

        buy_entry_price = reference_price - self.params["buy_threshold_offset"]
        sell_entry_price = reference_price + self.params["sell_threshold_offset"]

        buy_exit_price = reference_price - self.params["exit_offset"]
        sell_exit_price = reference_price + self.params["exit_offset"]

        if best_ask < buy_entry_price:  
            buy_deviation = buy_entry_price - best_ask
            target_size = self._calculate_order_size(buy_deviation)
            available_capacity = position_limit - current_position
            qty_to_buy = min(target_size, available_capacity)

            if qty_to_buy > 0:
                entry_order_price = int(best_ask)
                self.buy(entry_order_price, qty_to_buy)

        elif best_bid > sell_entry_price: 
            sell_deviation = best_bid - sell_entry_price
            target_size = self._calculate_order_size(sell_deviation)
            available_capacity = position_limit + current_position
            qty_to_sell = min(target_size, available_capacity)

            if qty_to_sell > 0:
                entry_order_price = int(best_bid)
                self.sell(entry_order_price, qty_to_sell)

        elif current_position > 0: 
            if best_bid > sell_exit_price:
                exit_ratio = self.params["partial_exit_ratio"]
                if best_bid >= reference_price:
                    exit_ratio = 1.0
                exit_size = max(1, int(current_position * exit_ratio))
                exit_size = min(exit_size, current_position)
                exit_order_price = int(best_bid)
                self.sell(exit_order_price, exit_size)

        elif current_position < 0:  
            if best_ask < buy_exit_price:
                exit_ratio = self.params["partial_exit_ratio"]
                if best_ask <= reference_price: 
                    exit_ratio = 1.0
                exit_size = max(1, int(abs(current_position) * exit_ratio))
                exit_size = min(exit_size, abs(current_position))
                exit_order_price = int(best_ask)
                self.buy(exit_order_price, exit_size)  

        elif current_position > 0 and best_ask < (buy_entry_price - self.params["tier1_threshold"]):
            additional_deviation = (buy_entry_price - self.params["tier1_threshold"]) - best_ask
            additional_size = self._calculate_order_size(additional_deviation)
            available_capacity = position_limit - current_position
            qty_to_add = min(additional_size, available_capacity)

            if qty_to_add > 0:
                scale_order_price = int(best_ask)
                self.buy(scale_order_price, qty_to_add)

        elif current_position < 0 and best_bid > (sell_entry_price + self.params["tier1_threshold"]):
            additional_deviation = best_bid - (sell_entry_price + self.params["tier1_threshold"])
            additional_size = self._calculate_order_size(additional_deviation)
            available_capacity = position_limit + current_position  
            qty_to_add = min(additional_size, available_capacity)

            if qty_to_add > 0:
                scale_order_price = int(best_bid)
                self.sell(scale_order_price, qty_to_add)

    def act(self, state: TradingState) -> None:
        if self.symbol not in state.order_depths:
            return
            
        order_depth = state.order_depths[self.symbol]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return
            
        current_position = state.position.get(self.symbol, 0)
        obs = state.observations.conversionObservations.get(self.symbol, None)
        
        if obs is None:
            return
            
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        
        if obs.sunlightIndex < 45:
            if self.last_s_index < obs.sunlightIndex:
                self.buy(round(buy_orders[0][0]), - abs(-75 - current_position))
            elif self.last_s_index > obs.sunlightIndex:
                self.sell(round(sell_orders[0][0]), -abs(75 - current_position))
        elif obs.sunlightIndex > 45:
            self._apply_mean_reversion(state)
        
        self.last_s_index = obs.sunlightIndex
    
    def save(self) -> JSON:
        return {
            "last_s_index": self.last_s_index,
            "params": self.params
        }
    
    def load(self, data: JSON) -> None:
        if data is not None:
            self.last_s_index = data.get("last_s_index", 100)
            if "params" in data:
                self.params = data["params"]

class MarketMakingStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)

        self.window = deque()
        self.window_size = 10

    def get_true_value(self, state: TradingState) -> int:
        raise NotImplementedError()

    def act(self, state: TradingState) -> None:
        true_value = self.get_true_value(state)

        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        self.window.append(abs(position) == self.limit)
        if len(self.window) > self.window_size:
            self.window.popleft()

        soft_liquidate = len(self.window) == self.window_size and sum(self.window) >= self.window_size / 2 and self.window[-1]
        hard_liquidate = len(self.window) == self.window_size and all(self.window)

        max_buy_price = true_value - 1 if position > self.limit * 0.5 else true_value
        min_sell_price = true_value + 1 if position < self.limit * -0.5 else true_value

        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity

        if to_buy > 0 and hard_liquidate:
            quantity = to_buy // 2
            self.buy(true_value, quantity)
            to_buy -= quantity

        if to_buy > 0 and soft_liquidate:
            quantity = to_buy // 2
            self.buy(true_value - 2, quantity)
            to_buy -= quantity

        if to_buy > 0:
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
            price = min(max_buy_price, popular_buy_price + 1)
            self.buy(price, to_buy)

        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity

        if to_sell > 0 and hard_liquidate:
            quantity = to_sell // 2
            self.sell(true_value, quantity)
            to_sell -= quantity

        if to_sell > 0 and soft_liquidate:
            quantity = to_sell // 2
            self.sell(true_value + 2, quantity)
            to_sell -= quantity

        if to_sell > 0:
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
            price = max(min_sell_price, popular_sell_price - 1)
            self.sell(price, to_sell)

    def save(self) -> JSON:
        return list(self.window)

    def load(self, data: JSON) -> None:
        self.window = deque(data)

class RainforestResinStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        return 10_000

class KelpStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

        return round((popular_buy_price + popular_sell_price) / 2)

class SquidInkStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.threshold = 5
        self.z_threshold = 2.0
        self.position_limit = limit
        self.data = {"highest_best_bid": None, "lowest_best_bid": None, "mid_prices": []}
        
    def act(self, state: TradingState) -> None:
        highest_best_bid = self.data.get("highest_best_bid")
        lowest_best_bid = self.data.get("lowest_best_bid")
        mid_prices = self.data.get("mid_prices", [])
        
        order_depth = state.order_depths[self.symbol]
        bids = order_depth.buy_orders
        asks = order_depth.sell_orders
        position = state.position.get(self.symbol, 0)
        
        if not bids or not asks:
            return
        
        best_bid = max(bids.keys())
        best_ask = min(asks.keys())
        mid_price = (best_bid + best_ask) / 2.0
        
        mid_prices.append(mid_price)
        if len(mid_prices) > 50:
            mid_prices = mid_prices[-50:]
        
        if highest_best_bid is None:
            highest_best_bid = best_bid
            lowest_best_bid = best_bid
        else:
            highest_best_bid = max(highest_best_bid, best_bid)
            lowest_best_bid = min(lowest_best_bid, best_bid)
        
        z_score = 0
        if len(mid_prices) >= 20:
            prices = np.array(mid_prices[-20:])
            mean_price = np.mean(prices)
            std_price = np.std(prices)
            if std_price > 0:
                z_score = (mid_price - mean_price) / std_price
        
        if best_bid < highest_best_bid - self.threshold:
            buy_size = min(20, self.position_limit - position)
            if buy_size > 0:
                self.buy(best_bid, buy_size)
                
        if best_bid > lowest_best_bid + self.threshold:
            sell_size = min(20, self.position_limit + position)
            if sell_size > 0:
                self.sell(best_ask, sell_size)
               
        if z_score < -self.z_threshold:
            buy_size = min(10, self.position_limit - position)
            if buy_size > 0:
                self.buy(best_bid, buy_size)
                
        if z_score > self.z_threshold:
            sell_size = min(10, self.position_limit + position)
            if sell_size > 0:
                self.sell(best_ask, sell_size)
               
        self.data["highest_best_bid"] = highest_best_bid
        self.data["lowest_best_bid"] = lowest_best_bid
        self.data["mid_prices"] = mid_prices
    
    def save(self) -> JSON:
        return self.data
    
    def load(self, data: JSON) -> None:
        if data is not None:
            self.data = data

class JAMSStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.params = {
            "reference_price": 6453.30,
            "buy_threshold_offset": 10.0,
            "sell_threshold_offset": 10.0,
            "exit_offset": 1.0,
            "position_limit": limit,
            "min_order_size": 10,
            "scaling_factor": 5.0,
            "tier1_threshold": 1.0,
            "tier2_threshold": 2.0,
            "tier3_threshold": 3.0,
            "tier1_size": 50,
            "tier2_size": 150,
            "tier3_size": 250,
            "partial_exit_ratio": 0.5
        }
        
    def _calculate_order_size(self, deviation_amount):
        """Calculates order size based on deviation tiers."""
        if deviation_amount >= self.params["tier3_threshold"]:
            return self.params["tier3_size"]
        elif deviation_amount >= self.params["tier2_threshold"]:
            return self.params["tier2_size"]
        elif deviation_amount >= self.params["tier1_threshold"]:
            return self.params["tier1_size"]
        else:
            base_size = int(self.params["min_order_size"] +
                        (deviation_amount * self.params["scaling_factor"]))
            return max(self.params["min_order_size"], base_size)

    def act(self, state: TradingState) -> None:
        if self.symbol not in state.order_depths:
            return

        order_depth = state.order_depths[self.symbol]
        current_position = state.position.get(self.symbol, 0)
        position_limit = self.params["position_limit"]

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return
        
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())

        mid_price = (best_bid + best_ask) / 2
        reference_price = self.params["reference_price"]
        deviation = mid_price - reference_price

        buy_entry_price = reference_price - self.params["buy_threshold_offset"]
        sell_entry_price = reference_price + self.params["sell_threshold_offset"]

        buy_exit_price = reference_price - self.params["exit_offset"]
        sell_exit_price = reference_price + self.params["exit_offset"]

        if best_ask < buy_entry_price: 
            buy_deviation = buy_entry_price - best_ask
            target_size = self._calculate_order_size(buy_deviation)
            available_capacity = position_limit - current_position
            qty_to_buy = min(target_size, available_capacity)

            if qty_to_buy > 0:
                entry_order_price = int(best_ask)
                self.buy(entry_order_price, qty_to_buy)


        elif best_bid > sell_entry_price:
            sell_deviation = best_bid - sell_entry_price
            target_size = self._calculate_order_size(sell_deviation)
            available_capacity = position_limit + current_position
            qty_to_sell = min(target_size, available_capacity)

            if qty_to_sell > 0:
                entry_order_price = int(best_bid)
                self.sell(entry_order_price, qty_to_sell)

        elif current_position > 0: 
            if best_bid > sell_exit_price:
                exit_ratio = self.params["partial_exit_ratio"]
                if best_bid >= reference_price: 
                    exit_ratio = 1.0
                exit_size = max(1, int(current_position * exit_ratio))
                exit_size = min(exit_size, current_position)
                exit_order_price = int(best_bid)
                self.sell(exit_order_price, exit_size)
                
        elif current_position < 0: 
            if best_ask < buy_exit_price:
                exit_ratio = self.params["partial_exit_ratio"]
                if best_ask <= reference_price: 
                    exit_ratio = 1.0
                exit_size = max(1, int(abs(current_position) * exit_ratio))
                exit_size = min(exit_size, abs(current_position))
                exit_order_price = int(best_ask)
                self.buy(exit_order_price, exit_size)
                
        elif current_position > 0 and best_ask < (buy_entry_price - self.params["tier1_threshold"]):
            additional_deviation = (buy_entry_price - self.params["tier1_threshold"]) - best_ask
            additional_size = self._calculate_order_size(additional_deviation)
            available_capacity = position_limit - current_position
            qty_to_add = min(additional_size, available_capacity)

            if qty_to_add > 0:
                scale_order_price = int(best_ask)
                self.buy(scale_order_price, qty_to_add)
                
        elif current_position < 0 and best_bid > (sell_entry_price + self.params["tier1_threshold"]):
            additional_deviation = best_bid - (sell_entry_price + self.params["tier1_threshold"])
            additional_size = self._calculate_order_size(additional_deviation)
            available_capacity = position_limit + current_position
            qty_to_add = min(additional_size, available_capacity)

            if qty_to_add > 0:
                scale_order_price = int(best_bid)
                self.sell(scale_order_price, qty_to_add)
                
    def save(self) -> JSON:
        return self.params
    
    def load(self, data: JSON) -> None:
        if data is not None:
            self.params = data
            
class PicnicBasket2Strategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.params = {
            "reference_price": 30141,
            "buy_threshold_offset": 3.0,
            "sell_threshold_offset": 3.0,
            "exit_offset": 10.0,
            "position_limit": limit,
            "min_order_size": 5,
            "scaling_factor": 2.0,
            "tier1_threshold": 6.0,
            "tier2_threshold": 12.0,
            "tier3_threshold": 18.0,
            "tier1_size": 10,
            "tier2_size": 25,
            "tier3_size": 50,
            "partial_exit_ratio": 0.5
        }
        

    def _calculate_order_size(self, deviation_amount):
        """Calculates order size based on deviation tiers."""
        if deviation_amount >= self.params["tier3_threshold"]:
            return self.params["tier3_size"]
        elif deviation_amount >= self.params["tier2_threshold"]:
            return self.params["tier2_size"]
        elif deviation_amount >= self.params["tier1_threshold"]:
            return self.params["tier1_size"]
        else:
            base_size = int(self.params["min_order_size"] +
                        (deviation_amount * self.params["scaling_factor"]))
            return max(self.params["min_order_size"], base_size)

    def act(self, state: TradingState) -> None:

        order_depth = state.order_depths[self.symbol]
        current_position = state.position.get(self.symbol, 0)
        position_limit = self.params["position_limit"]

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())

        mid_price = (best_bid + best_ask) / 2
        reference_price = self.params["reference_price"]
        deviation = mid_price - reference_price

        buy_entry_price = reference_price - self.params["buy_threshold_offset"]
        sell_entry_price = reference_price + self.params["sell_threshold_offset"]

        buy_exit_price = reference_price - self.params["exit_offset"]
        sell_exit_price = reference_price + self.params["exit_offset"]

        if best_ask < buy_entry_price:  
            buy_deviation = buy_entry_price - best_ask
            target_size = self._calculate_order_size(buy_deviation)
            available_capacity = position_limit - current_position
            qty_to_buy = min(target_size, available_capacity)

            if qty_to_buy > 0:
                entry_order_price = int(best_ask)
                self.buy(entry_order_price, qty_to_buy)
                
        elif best_bid > sell_entry_price: 
            sell_deviation = best_bid - sell_entry_price
            target_size = self._calculate_order_size(sell_deviation)
            available_capacity = position_limit + current_position
            qty_to_sell = min(target_size, available_capacity)

            if qty_to_sell > 0:
                entry_order_price = int(best_bid)
                self.sell(entry_order_price, qty_to_sell)
                
        # 2. Exit Logic
        elif current_position > 0:  # Long position exit
            if best_bid > sell_exit_price:
                exit_ratio = self.params["partial_exit_ratio"]
                if best_bid >= reference_price:  # Exit fully if back at/above reference
                    exit_ratio = 1.0
                exit_size = max(1, int(current_position * exit_ratio))
                exit_size = min(exit_size, current_position)
                exit_order_price = int(best_bid)
                self.sell(exit_order_price, exit_size)
                
        elif current_position < 0:  # Short position exit
            if best_ask < buy_exit_price:
                exit_ratio = self.params["partial_exit_ratio"]
                if best_ask <= reference_price:  # Exit fully if back at/below reference
                    exit_ratio = 1.0
                exit_size = max(1, int(abs(current_position) * exit_ratio))
                exit_size = min(exit_size, abs(current_position))
                exit_order_price = int(best_ask)
                self.buy(exit_order_price, exit_size)
                
        # 3. Position Scaling Logic
        elif current_position > 0 and best_ask < (buy_entry_price - self.params["tier1_threshold"]):
            additional_deviation = (buy_entry_price - self.params["tier1_threshold"]) - best_ask
            additional_size = self._calculate_order_size(additional_deviation)
            available_capacity = position_limit - current_position
            qty_to_add = min(additional_size, available_capacity)

            if qty_to_add > 0:
                scale_order_price = int(best_ask)
                self.buy(scale_order_price, qty_to_add)
                
        elif current_position < 0 and best_bid > (sell_entry_price + self.params["tier1_threshold"]):
            additional_deviation = best_bid - (sell_entry_price + self.params["tier1_threshold"])
            additional_size = self._calculate_order_size(additional_deviation)
            available_capacity = position_limit + current_position
            qty_to_add = min(additional_size, available_capacity)

            if qty_to_add > 0:
                scale_order_price = int(best_bid)
                self.sell(scale_order_price, qty_to_add)
                
    def save(self) -> JSON:
        return self.params
    
    def load(self, data: JSON) -> None:
        if data is not None:
            self.params = data

class Rock_MeanStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.params = {
            "reference_price": 10205,
            "buy_threshold_offset": 2.0,
            "sell_threshold_offset": 2.0,
            "exit_offset": 5.0,
            "position_limit": limit,
            "min_order_size": 5,
            "scaling_factor": 2.0,
            "tier1_threshold": 6.0,
            "tier2_threshold": 12.0,
            "tier3_threshold": 18.0,
            "tier1_size": 100,
            "tier2_size": 250,
            "tier3_size": 400,
            "partial_exit_ratio": 0.8
        }
        

    def _calculate_order_size(self, deviation_amount):
        """Calculates order size based on deviation tiers."""
        if deviation_amount >= self.params["tier3_threshold"]:
            return self.params["tier3_size"]
        elif deviation_amount >= self.params["tier2_threshold"]:
            return self.params["tier2_size"]
        elif deviation_amount >= self.params["tier1_threshold"]:
            return self.params["tier1_size"]
        else:
            base_size = int(self.params["min_order_size"] +
                        (deviation_amount * self.params["scaling_factor"]))
            return max(self.params["min_order_size"], base_size)

    def act(self, state: TradingState) -> None:

        order_depth = state.order_depths[self.symbol]
        current_position = state.position.get(self.symbol, 0)
        position_limit = self.params["position_limit"]

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())

        mid_price = (best_bid + best_ask) / 2
            
        reference_price = self.params["reference_price"]
        deviation = mid_price - reference_price

        buy_entry_price = reference_price - self.params["buy_threshold_offset"]
        sell_entry_price = reference_price + self.params["sell_threshold_offset"]

        buy_exit_price = reference_price - self.params["exit_offset"]
        sell_exit_price = reference_price + self.params["exit_offset"]

        if best_ask < buy_entry_price:  
            buy_deviation = buy_entry_price - best_ask
            target_size = self._calculate_order_size(buy_deviation)
            available_capacity = position_limit - current_position
            qty_to_buy = min(target_size, available_capacity)

            if qty_to_buy > 0:
                entry_order_price = int(best_ask)
                self.buy(entry_order_price, qty_to_buy)
                
        elif best_bid > sell_entry_price: 
            sell_deviation = best_bid - sell_entry_price
            target_size = self._calculate_order_size(sell_deviation)
            available_capacity = position_limit + current_position
            qty_to_sell = min(target_size, available_capacity)

            if qty_to_sell > 0:
                entry_order_price = int(best_bid)
                self.sell(entry_order_price, qty_to_sell)
                
        # 2. Exit Logic
        elif current_position > 0:  # Long position exit
            if best_bid > sell_exit_price:
                exit_ratio = self.params["partial_exit_ratio"]
                if best_bid >= reference_price:  # Exit fully if back at/above reference
                    exit_ratio = 1.0
                exit_size = max(1, int(current_position * exit_ratio))
                exit_size = min(exit_size, current_position)
                exit_order_price = int(best_bid)
                self.sell(exit_order_price, exit_size)
                
        elif current_position < 0:  # Short position exit
            if best_ask < buy_exit_price:
                exit_ratio = self.params["partial_exit_ratio"]
                if best_ask <= reference_price:  # Exit fully if back at/below reference
                    exit_ratio = 1.0
                exit_size = max(1, int(abs(current_position) * exit_ratio))
                exit_size = min(exit_size, abs(current_position))
                exit_order_price = int(best_ask)
                self.buy(exit_order_price, exit_size)
                
        # 3. Position Scaling Logic
        elif current_position > 0 and best_ask < (buy_entry_price - self.params["tier1_threshold"]):
            additional_deviation = (buy_entry_price - self.params["tier1_threshold"]) - best_ask
            additional_size = self._calculate_order_size(additional_deviation)
            available_capacity = position_limit - current_position
            qty_to_add = min(additional_size, available_capacity)

            if qty_to_add > 0:
                scale_order_price = int(best_ask)
                self.buy(scale_order_price, qty_to_add)
                
        elif current_position < 0 and best_bid > (sell_entry_price + self.params["tier1_threshold"]):
            additional_deviation = best_bid - (sell_entry_price + self.params["tier1_threshold"])
            additional_size = self._calculate_order_size(additional_deviation)
            available_capacity = position_limit + current_position
            qty_to_add = min(additional_size, available_capacity)

            if qty_to_add > 0:
                scale_order_price = int(best_bid)
                self.sell(scale_order_price, qty_to_add)
                
    def save(self) -> JSON:
        return self.params
    
    def load(self, data: JSON) -> None:
        if data is not None:
            self.params = data

class MirrorVolcanicRockStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        """
        Initialize the strategy.

        :param symbol: The symbol for which this strategy is applied.
        :param limit: The position limit for this symbol.
        """
        super().__init__(symbol, limit)
        self.target_symbol = "VOLCANIC_ROCK"  # Symbol to mirror

    def act(self, state: TradingState) -> None:
        """
        Act based on the position of the target symbol.

        :param state: The current trading state.
        """
        # Get the current position of the target symbol
        target_position = state.position.get(self.target_symbol, 0)
        current_position = state.position.get(self.symbol, 0)

        # Calculate the difference between the target and current positions
        position_difference = target_position - current_position

        # Get the order depth for the symbol
        order_depth = state.order_depths.get(self.symbol, None)
        if not order_depth:
            return  # Skip if there is no order depth for the symbol

        # Get the best bid and best ask prices
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        if position_difference > 0 and best_ask is not None:
            # Buy to match the target position at the best ask price
            quantity_to_buy = min(position_difference, self.limit - current_position)
            self.buy(best_ask, quantity_to_buy)

        elif position_difference < 0 and best_bid is not None:
            # Sell to match the target position at the best bid price
            quantity_to_sell = min(abs(position_difference), self.limit + current_position)
            self.sell(best_bid, quantity_to_sell)
                             
class BasketStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.data = {
            "spread_history": [],
            "prev_zscore": 0,
            "clear_flag": False,
            "curr_avg": 0,
        }
        self.basket_products = [
            Product.CHOCOLATE,
            Product.STRAWBERRIES,
            Product.ROSES,
            Product.DJEMBES
        ]
        self.params = BASKET_PARAMS[Product.SPREAD]
        self.position_limit = limit
        self.external_orders = {}
        
    def get_swmid(self, order_depth) -> float:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 0
            
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (
            best_bid_vol + best_ask_vol
        )
        
    def get_synthetic_basket_order_depth(
        self, order_depths: dict[Symbol, OrderDepth]
    ) -> OrderDepth:
        synthetic_order_price = OrderDepth()
        
        # Get all component prices
        component_prices = {}
        for product in self.basket_products:
            if product not in order_depths or not order_depths[product].buy_orders or not order_depths[product].sell_orders:
                return synthetic_order_price  # Return empty if any product is missing
                
            component_prices[product] = {
                "best_bid": max(order_depths[product].buy_orders.keys()),
                "best_ask": min(order_depths[product].sell_orders.keys()),
                "bid_volume": order_depths[product].buy_orders[max(order_depths[product].buy_orders.keys())],
                "ask_volume": abs(order_depths[product].sell_orders[min(order_depths[product].sell_orders.keys())])
            }
        
        # Calculate implied basket prices
        implied_bid = sum(component_prices[product]["best_bid"] * BASKET_WEIGHTS[product] 
                          for product in self.basket_products)
        implied_ask = sum(component_prices[product]["best_ask"] * BASKET_WEIGHTS[product] 
                          for product in self.basket_products)
        
        # Calculate max volumes
        bid_volumes = [component_prices[product]["bid_volume"] // BASKET_WEIGHTS[product] 
                      for product in self.basket_products]
        ask_volumes = [component_prices[product]["ask_volume"] // BASKET_WEIGHTS[product] 
                      for product in self.basket_products]
                      
        implied_bid_volume = min(bid_volumes) if bid_volumes else 0
        implied_ask_volume = min(ask_volumes) if ask_volumes else 0
        
        if implied_bid_volume > 0:
            synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume
        
        if implied_ask_volume > 0:
            synthetic_order_price.sell_orders[implied_ask] = -implied_ask_volume
            
        return synthetic_order_price
        
    def execute_spread_orders(
        self,
        target_position: int,
        basket_position: int,
        order_depths: Dict[str, OrderDepth],
    ):
        if target_position == basket_position:
            return None
            
        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[Product.GIFT_BASKET]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths)
        
        # Check if we have sufficient data
        if (not basket_order_depth.buy_orders or not basket_order_depth.sell_orders or
            not synthetic_order_depth.buy_orders or not synthetic_order_depth.sell_orders):
            return None
            
        result = {product: [] for product in self.basket_products}
        result[Product.GIFT_BASKET] = []
        
        if target_position > basket_position:  # Need to buy baskets
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])
            
            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(synthetic_order_depth.buy_orders[synthetic_bid_price])
            
            execute_volume = min(basket_ask_volume, synthetic_bid_volume, target_quantity)
            
            if execute_volume > 0:
                # Buy the basket
                result[Product.GIFT_BASKET].append(Order(Product.GIFT_BASKET, basket_ask_price, execute_volume))
                
                # Sell the components
                for product in self.basket_products:
                    component_price = max(order_depths[product].buy_orders.keys())
                    result[product].append(Order(product, component_price, -execute_volume * BASKET_WEIGHTS[product]))
                    
        else:  # Need to sell baskets
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])
            
            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(synthetic_order_depth.sell_orders[synthetic_ask_price])
            
            execute_volume = min(basket_bid_volume, synthetic_ask_volume, target_quantity)
            
            if execute_volume > 0:
                # Sell the basket
                result[Product.GIFT_BASKET].append(Order(Product.GIFT_BASKET, basket_bid_price, -execute_volume))
                
                # Buy the components
                for product in self.basket_products:
                    component_price = min(order_depths[product].sell_orders.keys())
                    result[product].append(Order(product, component_price, execute_volume * BASKET_WEIGHTS[product]))
                    
        return result
        
    def act(self, state: TradingState) -> None:
        # Need to handle basket position and all component positions
        basket_position = state.position.get(Product.GIFT_BASKET, 0)
        
        # Skip if basket data isn't available
        if Product.GIFT_BASKET not in state.order_depths:
            return
            
        # Check if all components are available
        for product in self.basket_products:
            if product not in state.order_depths:
                return
        
        basket_order_depth = state.order_depths[Product.GIFT_BASKET]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(state.order_depths)
        
        # Skip if either order book is empty
        if (not basket_order_depth.buy_orders or not basket_order_depth.sell_orders or
            not synthetic_order_depth.buy_orders or not synthetic_order_depth.sell_orders):
            return
            
        # Calculate spread
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        spread = basket_swmid - synthetic_swmid
        
        # Update history
        self.data["spread_history"].append(spread)
        spread_history = self.data["spread_history"]
        
        # Ensure history isn't too long
        if len(spread_history) > self.params["spread_std_window"]:
            spread_history.pop(0)
            
        # Skip if not enough history
        if len(spread_history) < self.params["spread_std_window"]:
            return
            
        # Calculate z-score
        current_std = self.params["current_std"]
        spread_std = np.std(spread_history[-current_std:])
        if spread_std == 0:
            return
            
        z_score_thresh = (
            np.mean(spread_history) - self.params["default_spread_mean"]
        ) / np.std(spread_history)
        
        zscore = (spread - self.params["default_spread_mean"]) / spread_std
        
        # Determine trading thresholds
        if z_score_thresh > 0:
            upper = z_score_thresh + self.params["zscore_threshold"] 
            lower = z_score_thresh - self.params["zscore_threshold"] 
        else:
            upper = z_score_thresh + self.params["zscore_threshold"] 
            lower = z_score_thresh - self.params["zscore_threshold"] 
            
        # Execute trades based on z-score signals
        if zscore >= upper:
            if basket_position != -self.params["target_position"]:
                spread_orders = self.execute_spread_orders(
                    -self.params["target_position"],
                    basket_position,
                    state.order_depths
                )
                if spread_orders:
                    # Add the orders
                    for product, orders in spread_orders.items():
                        for order in orders:
                            if product == self.symbol:
                                self.orders.append(order)
                            else:
                                # Need to handle multi-symbol orders differently
                               
                                # Store these orders to be processed by Trader class
                                if product not in self.external_orders:
                                    self.external_orders[product] = []
                                self.external_orders[product].append(order)
        
        elif zscore <= lower:
            if basket_position != self.params["target_position"]:
                spread_orders = self.execute_spread_orders(
                    self.params["target_position"],
                    basket_position,
                    state.order_depths
                )
                if spread_orders:
                    # Add the orders
                    for product, orders in spread_orders.items():
                        for order in orders:
                            if product == self.symbol:
                                self.orders.append(order)
                            else:
                                # Need to handle multi-symbol orders differently
                                
                                # Store these orders to be processed by Trader class
                                if product not in self.external_orders:
                                    self.external_orders[product] = []
                                self.external_orders[product].append(order)
        
        # Update data
        self.data["prev_zscore"] = zscore
        
    def save(self) -> JSON:
        return self.data
        
    def load(self, data: JSON) -> None:
        if data is not None:
            self.data = data
            # Make sure spread_history exists
            if "spread_history" not in self.data:
                self.data["spread_history"] = []
                
class MaxLongPositionStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        """
        Initialize the strategy.

        :param symbol: The symbol for which this strategy is applied.
        :param limit: The maximum position limit for this symbol.
        """
        super().__init__(symbol, limit)

    def act(self, state: TradingState) -> None:
        """
        Act to ensure the position reaches the maximum long limit.

        :param state: The current trading state.
        """
        # Get the current position for the symbol
        current_position = state.position.get(self.symbol, 0)

        # Get the order depth for the symbol
        order_depth = state.order_depths.get(self.symbol, None)
        if not order_depth or not order_depth.sell_orders:
            return  # Skip if there are no sell orders

        # Get the best ask price and volume
        best_ask = min(order_depth.sell_orders.keys())
        best_ask_volume = abs(order_depth.sell_orders[best_ask])

        # Calculate the quantity needed to reach the maximum long position
        quantity_to_buy = min(self.limit - current_position, best_ask_volume)

        # Place a buy order if there is room to increase the position
        if quantity_to_buy > 0:
            self.buy(best_ask, quantity_to_buy)

class Rock_Simple_MeanStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.params = {
            "reference_price": 10000,  # Default reference price for the first 100 periods
            "buy_threshold_offset": 3.0,
            "sell_threshold_offset": 3.0,
            "exit_offset": 10.0,
            "position_limit": limit,
            "min_order_size": 5,
            "scaling_factor": 2.0,
            "tier1_threshold": 3.0,
            "tier2_threshold": 6.0,
            "tier3_threshold": 9.0,
            "tier1_size": 100,
            "tier2_size": 250,
            "tier3_size": 400,
            "partial_exit_ratio": 0.75,
        }
        self.mid_prices = []  # Store mid-prices for SMA calculation

    def _calculate_order_size(self, deviation_amount):
        """Calculates order size based on deviation tiers."""
        if deviation_amount >= self.params["tier3_threshold"]:
            return self.params["tier3_size"]
        elif deviation_amount >= self.params["tier2_threshold"]:
            return self.params["tier2_size"]
        elif deviation_amount >= self.params["tier1_threshold"]:
            return self.params["tier1_size"]
        else:
            base_size = int(self.params["min_order_size"] +
                            (deviation_amount * self.params["scaling_factor"]))
            return max(self.params["min_order_size"], base_size)

    def _calculate_sma(self):
        """Calculates the Simple Moving Average (SMA) of the last 100 mid-prices."""
        if len(self.mid_prices) >= 100:
            return sum(self.mid_prices[-100:]) / 100
        return self.params["reference_price"]  # Use default reference price before 100 periods

    def act(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        current_position = state.position.get(self.symbol, 0)
        position_limit = self.params["position_limit"]

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return

        # Calculate mid-price
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2

        # Update mid-price history
        self.mid_prices.append(mid_price)
        if len(self.mid_prices) > 200:  # Keep the list manageable
            self.mid_prices.pop(0)

        # Calculate reference price (SMA of last 100 periods or default before 100 periods)
        reference_price = self._calculate_sma()
        deviation = mid_price - reference_price

        buy_entry_price = reference_price - self.params["buy_threshold_offset"]
        sell_entry_price = reference_price + self.params["sell_threshold_offset"]

        buy_exit_price = reference_price - self.params["exit_offset"]
        sell_exit_price = reference_price + self.params["exit_offset"]

        # Entry logic
        if best_ask < buy_entry_price:
            buy_deviation = buy_entry_price - best_ask
            target_size = self._calculate_order_size(buy_deviation)
            available_capacity = position_limit - current_position
            qty_to_buy = min(target_size, available_capacity)

            if qty_to_buy > 0:
                entry_order_price = int(best_ask)
                self.buy(entry_order_price, qty_to_buy)

        elif best_bid > sell_entry_price:
            sell_deviation = best_bid - sell_entry_price
            target_size = self._calculate_order_size(sell_deviation)
            available_capacity = position_limit + current_position
            qty_to_sell = min(target_size, available_capacity)

            if qty_to_sell > 0:
                entry_order_price = int(best_bid)
                self.sell(entry_order_price, qty_to_sell)

        # Exit logic
        elif current_position > 0:  # Long position exit
            if best_bid > sell_exit_price:
                exit_ratio = self.params["partial_exit_ratio"]
                if best_bid >= reference_price:  # Exit fully if back at/above reference
                    exit_ratio = 1.0
                exit_size = max(1, int(current_position * exit_ratio))
                exit_size = min(exit_size, current_position)
                exit_order_price = int(best_bid)
                self.sell(exit_order_price, exit_size)

        elif current_position < 0:  # Short position exit
            if best_ask < buy_exit_price:
                exit_ratio = self.params["partial_exit_ratio"]
                if best_ask <= reference_price:  # Exit fully if back at/below reference
                    exit_ratio = 1.0
                exit_size = max(1, int(abs(current_position) * exit_ratio))
                exit_size = min(exit_size, abs(current_position))
                exit_order_price = int(best_ask)
                self.buy(exit_order_price, exit_size)

        # Position scaling logic
        elif current_position > 0 and best_ask < (buy_entry_price - self.params["tier1_threshold"]):
            additional_deviation = (buy_entry_price - self.params["tier1_threshold"]) - best_ask
            additional_size = self._calculate_order_size(additional_deviation)
            available_capacity = position_limit - current_position
            qty_to_add = min(additional_size, available_capacity)

            if qty_to_add > 0:
                scale_order_price = int(best_ask)
                self.buy(scale_order_price, qty_to_add)

        elif current_position < 0 and best_bid > (sell_entry_price + self.params["tier1_threshold"]):
            additional_deviation = best_bid - (sell_entry_price + self.params["tier1_threshold"])
            additional_size = self._calculate_order_size(additional_deviation)
            available_capacity = position_limit + current_position
            qty_to_add = min(additional_size, available_capacity)

            if qty_to_add > 0:
                scale_order_price = int(best_bid)
                self.sell(scale_order_price, qty_to_add)

    def save(self) -> JSON:
        return {"mid_prices": self.mid_prices, "params": self.params}

    def load(self, data: JSON) -> None:
        if data is not None:
            self.mid_prices = data.get("mid_prices", [])
            self.params = data.get("params", self.params)

class Rock_Step_MeanStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.params = {
            "reference_price": 1,  # Default reference price for the first 100 periods
            "buy_threshold_offset": 3.0,
            "sell_threshold_offset": 3.0,
            "exit_offset": 10.0,
            "position_limit": limit,
            "min_order_size": 5,
            "scaling_factor": 2.0,
            "tier1_threshold": 3.0,
            "tier2_threshold": 6.0,
            "tier3_threshold": 9.0,
            "tier1_size": 100,
            "tier2_size": 250,
            "tier3_size": 400,
            "partial_exit_ratio": 0.75,
        }
        self.mid_prices = []  # Store mid-prices for tracking
        self.reference_price = self.params["reference_price"]  # Initialize reference price

    def _calculate_order_size(self, deviation_amount):
        """Calculates order size based on deviation tiers."""
        if deviation_amount >= self.params["tier3_threshold"]:
            return self.params["tier3_size"]
        elif deviation_amount >= self.params["tier2_threshold"]:
            return self.params["tier2_size"]
        elif deviation_amount >= self.params["tier1_threshold"]:
            return self.params["tier1_size"]
        else:
            base_size = int(self.params["min_order_size"] +
                            (deviation_amount * self.params["scaling_factor"]))
            return max(self.params["min_order_size"], base_size)

    def act(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        current_position = state.position.get(self.symbol, 0)
        position_limit = self.params["position_limit"]

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return

        # Calculate mid-price
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2

        # Update reference price dynamically with change/2
        if self.mid_prices:
            if self.reference_price ==1 :
                self.reference_price = mid_price
            else: 
                last_mid_price = self.mid_prices[-1]
                price_change = mid_price - last_mid_price
                self.reference_price += price_change // 3

        # Update mid-price history
        self.mid_prices.append(mid_price)
        if len(self.mid_prices) > 200:  # Keep the list manageable
            self.mid_prices.pop(0)

        # Calculate deviation from the reference price
        deviation = mid_price - self.reference_price

        buy_entry_price = self.reference_price - self.params["buy_threshold_offset"]
        sell_entry_price = self.reference_price + self.params["sell_threshold_offset"]

        buy_exit_price = self.reference_price - self.params["exit_offset"]
        sell_exit_price = self.reference_price + self.params["exit_offset"]

        # Entry logic
        if best_ask < buy_entry_price:
            buy_deviation = buy_entry_price - best_ask
            target_size = self._calculate_order_size(buy_deviation)
            available_capacity = position_limit - current_position
            qty_to_buy = min(target_size, available_capacity)

            if qty_to_buy > 0:
                entry_order_price = int(best_ask)
                self.buy(entry_order_price, qty_to_buy)

        elif best_bid > sell_entry_price:
            sell_deviation = best_bid - sell_entry_price
            target_size = self._calculate_order_size(sell_deviation)
            available_capacity = position_limit + current_position
            qty_to_sell = min(target_size, available_capacity)

            if qty_to_sell > 0:
                entry_order_price = int(best_bid)
                self.sell(entry_order_price, qty_to_sell)

        # Exit logic
        elif current_position > 0:  # Long position exit
            if best_bid > sell_exit_price:
                exit_ratio = self.params["partial_exit_ratio"]
                if best_bid >= self.reference_price:  # Exit fully if back at/above reference
                    exit_ratio = 1.0
                exit_size = max(1, int(current_position * exit_ratio))
                exit_size = min(exit_size, current_position)
                exit_order_price = int(best_bid)
                self.sell(exit_order_price, exit_size)

        elif current_position < 0:  # Short position exit
            if best_ask < buy_exit_price:
                exit_ratio = self.params["partial_exit_ratio"]
                if best_ask <= self.reference_price:  # Exit fully if back at/below reference
                    exit_ratio = 1.0
                exit_size = max(1, int(abs(current_position) * exit_ratio))
                exit_size = min(exit_size, abs(current_position))
                exit_order_price = int(best_ask)
                self.buy(exit_order_price, exit_size)

        # Position scaling logic
        elif current_position > 0 and best_ask < (buy_entry_price - self.params["tier1_threshold"]):
            additional_deviation = (buy_entry_price - self.params["tier1_threshold"]) - best_ask
            additional_size = self._calculate_order_size(additional_deviation)
            available_capacity = position_limit - current_position
            qty_to_add = min(additional_size, available_capacity)

            if qty_to_add > 0:
                scale_order_price = int(best_ask)
                self.buy(scale_order_price, qty_to_add)

        elif current_position < 0 and best_bid > (sell_entry_price + self.params["tier1_threshold"]):
            additional_deviation = best_bid - (sell_entry_price + self.params["tier1_threshold"])
            additional_size = self._calculate_order_size(additional_deviation)
            available_capacity = position_limit + current_position
            qty_to_add = min(additional_size, available_capacity)

            if qty_to_add > 0:
                scale_order_price = int(best_bid)
                self.sell(scale_order_price, qty_to_add)

    def save(self) -> JSON:
        return {"mid_prices": self.mid_prices, "reference_price": self.reference_price, "params": self.params}

    def load(self, data: JSON) -> None:
        if data is not None:
            self.mid_prices = data.get("mid_prices", [])
            self.reference_price = data.get("reference_price", self.params["reference_price"])
            self.params = data.get("params", self.params)

class Rock_Hull_MeanStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.params = {
            "reference_price": 10000,  # Default reference price for the first 100 periods
            "buy_threshold_offset": 3.0,
            "sell_threshold_offset": 3.0,
            "exit_offset": 10.0,
            "position_limit": limit,
            "min_order_size": 5,
            "scaling_factor": 2.0,
            "tier1_threshold": 3.0,
            "tier2_threshold": 6.0,
            "tier3_threshold": 9.0,
            "tier1_size": 100,
            "tier2_size": 250,
            "tier3_size": 400,
            "partial_exit_ratio": 0.75,
        }
        self.mid_prices = []  # Store mid-prices for HMA calculation

    def _calculate_wma(self, prices: list[float], period: int) -> float:
        """Calculates the Weighted Moving Average (WMA) for a given period."""
        if len(prices) < period:
            return sum(prices) / len(prices)  # Fallback to SMA if not enough data
        weights = list(range(1, period + 1))
        weighted_sum = sum(p * w for p, w in zip(prices[-period:], weights))
        return weighted_sum / sum(weights)

    def _calculate_hma(self, period: int) -> float:
        """Calculates the Hull Moving Average (HMA) for a given period."""
        if len(self.mid_prices) < period:
            return sum(self.mid_prices) / len(self.mid_prices)  # Fallback to SMA if not enough data

        # Step 1: Calculate WMA for half the period and full period
        wma_half = self._calculate_wma(self.mid_prices, period // 2)
        wma_full = self._calculate_wma(self.mid_prices, period)

        # Step 2: Intermediate calculation
        intermediate = [2 * wma_half - wma_full]

        # Step 3: Final HMA
        hma_period = int(period**0.5)  # Square root of the period
        return self._calculate_wma(intermediate, hma_period)
    
    def _calculate_order_size(self, deviation_amount):
        """Calculates order size based on deviation tiers."""
        if deviation_amount >= self.params["tier3_threshold"]:
            return self.params["tier3_size"]
        elif deviation_amount >= self.params["tier2_threshold"]:
            return self.params["tier2_size"]
        elif deviation_amount >= self.params["tier1_threshold"]:
            return self.params["tier1_size"]
        else:
            base_size = int(self.params["min_order_size"] +
                            (deviation_amount * self.params["scaling_factor"]))
            return max(self.params["min_order_size"], base_size)

    def act(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        current_position = state.position.get(self.symbol, 0)
        position_limit = self.params["position_limit"]

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return

        # Calculate mid-price
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2

        # Update mid-price history
        self.mid_prices.append(mid_price)
        if len(self.mid_prices) > 200:  # Keep the list manageable
            self.mid_prices.pop(0)

        # Calculate reference price using HMA
        reference_price = self._calculate_hma(100)  # Use a period of 20 for HMA
        deviation = mid_price - reference_price

        buy_entry_price = reference_price - self.params["buy_threshold_offset"]
        sell_entry_price = reference_price + self.params["sell_threshold_offset"]

        buy_exit_price = reference_price - self.params["exit_offset"]
        sell_exit_price = reference_price + self.params["exit_offset"]

        # Entry logic
        if best_ask < buy_entry_price:
            buy_deviation = buy_entry_price - best_ask
            target_size = self._calculate_order_size(buy_deviation)
            available_capacity = position_limit - current_position
            qty_to_buy = min(target_size, available_capacity)

            if qty_to_buy > 0:
                entry_order_price = int(best_ask)
                self.buy(entry_order_price, qty_to_buy)

        elif best_bid > sell_entry_price:
            sell_deviation = best_bid - sell_entry_price
            target_size = self._calculate_order_size(sell_deviation)
            available_capacity = position_limit + current_position
            qty_to_sell = min(target_size, available_capacity)

            if qty_to_sell > 0:
                entry_order_price = int(best_bid)
                self.sell(entry_order_price, qty_to_sell)

        # Exit logic
        elif current_position > 0:  # Long position exit
            if best_bid > sell_exit_price:
                exit_ratio = self.params["partial_exit_ratio"]
                if best_bid >= reference_price:  # Exit fully if back at/above reference
                    exit_ratio = 1.0
                exit_size = max(1, int(current_position * exit_ratio))
                exit_size = min(exit_size, current_position)
                exit_order_price = int(best_bid)
                self.sell(exit_order_price, exit_size)

        elif current_position < 0:  # Short position exit
            if best_ask < buy_exit_price:
                exit_ratio = self.params["partial_exit_ratio"]
                if best_ask <= reference_price:  # Exit fully if back at/below reference
                    exit_ratio = 1.0
                exit_size = max(1, int(abs(current_position) * exit_ratio))
                exit_size = min(exit_size, abs(current_position))
                exit_order_price = int(best_ask)
                self.buy(exit_order_price, exit_size)

        # Position scaling logic
        elif current_position > 0 and best_ask < (buy_entry_price - self.params["tier1_threshold"]):
            additional_deviation = (buy_entry_price - self.params["tier1_threshold"]) - best_ask
            additional_size = self._calculate_order_size(additional_deviation)
            available_capacity = position_limit - current_position
            qty_to_add = min(additional_size, available_capacity)

            if qty_to_add > 0:
                scale_order_price = int(best_ask)
                self.buy(scale_order_price, qty_to_add)

        elif current_position < 0 and best_bid > (sell_entry_price + self.params["tier1_threshold"]):
            additional_deviation = best_bid - (sell_entry_price + self.params["tier1_threshold"])
            additional_size = self._calculate_order_size(additional_deviation)
            available_capacity = position_limit + current_position
            qty_to_add = min(additional_size, available_capacity)

            if qty_to_add > 0:
                scale_order_price = int(best_bid)
                self.sell(scale_order_price, qty_to_add)

    def save(self) -> JSON:
        return {"mid_prices": self.mid_prices, "params": self.params}

    def load(self, data: JSON) -> None:
        if data is not None:
            self.mid_prices = data.get("mid_prices", [])
            self.params = data.get("params", self.params)

class Rock_Exp_MeanStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.params = {
            "reference_price": 10000,  # Default reference price for the first 100 periods
            "buy_threshold_offset": 3.0,
            "sell_threshold_offset": 3.0,
            "exit_offset": 20.0,
            "position_limit": limit,
            "min_order_size": 5,
            "scaling_factor": 2.0,
            "tier1_threshold": 6.0,
            "tier2_threshold": 12.0,
            "tier3_threshold": 18.0,
            "tier1_size": 100,
            "tier2_size": 250,
            "tier3_size": 400,
            "partial_exit_ratio": 0.5,
        }
        self.mid_prices = []  # Store mid-prices for EMA calculation
        self.ema_values = {}  # Store EMA values for different periods

    def _calculate_ema(self, period: int, new_price: float) -> float:
        """
        Calculates the Exponential Moving Average (EMA) for the given period.
        :param period: The period for the EMA.
        :param new_price: The latest price to include in the EMA calculation.
        :return: The updated EMA value.
        """
        if period not in self.ema_values:
            # Initialize EMA with the first price if not already calculated
            self.ema_values[period] = new_price
            return new_price

        # Calculate the smoothing factor
        alpha = 2 / (period + 1)
        # Update the EMA value
        self.ema_values[period] = alpha * new_price + (1 - alpha) * self.ema_values[period]
        return self.ema_values[period]

    def _get_reference_price(self) -> float:
        """
        Determines the reference price based on EMA conditions.
        If EMA(10) > EMA(30), the reference price is EMA(100) + 50.
        Otherwise, the reference price is EMA(100) - 50.
        """
        ema_10 = self._calculate_ema(10, self.mid_prices[-1])
        ema_30 = self._calculate_ema(30, self.mid_prices[-1])
        ema_100 = self._calculate_ema(100, self.mid_prices[-1])

        if ema_10 > ema_30:
            return ema_100
        else:
            return ema_100

    def act(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        current_position = state.position.get(self.symbol, 0)
        position_limit = self.params["position_limit"]

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return

        # Calculate mid-price
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2

        # Update mid-price history
        self.mid_prices.append(mid_price)
        if len(self.mid_prices) > 200:  # Keep the list manageable
            self.mid_prices.pop(0)

        # Calculate reference price based on EMA logic
        reference_price = self._get_reference_price()
        deviation = mid_price - reference_price

        buy_entry_price = reference_price - self.params["buy_threshold_offset"]
        sell_entry_price = reference_price + self.params["sell_threshold_offset"]

        buy_exit_price = reference_price - self.params["exit_offset"]
        sell_exit_price = reference_price + self.params["exit_offset"]

        # Entry logic
        if best_ask < buy_entry_price:
            buy_deviation = buy_entry_price - best_ask
            target_size = self._calculate_order_size(buy_deviation)
            available_capacity = position_limit - current_position
            qty_to_buy = min(target_size, available_capacity)

            if qty_to_buy > 0:
                entry_order_price = int(best_ask)
                self.buy(entry_order_price, qty_to_buy)

        elif best_bid > sell_entry_price:
            sell_deviation = best_bid - sell_entry_price
            target_size = self._calculate_order_size(sell_deviation)
            available_capacity = position_limit + current_position
            qty_to_sell = min(target_size, available_capacity)

            if qty_to_sell > 0:
                entry_order_price = int(best_bid)
                self.sell(entry_order_price, qty_to_sell)

        # Exit logic
        elif current_position > 0:  # Long position exit
            if best_bid > sell_exit_price:
                exit_ratio = self.params["partial_exit_ratio"]
                if best_bid >= reference_price:  # Exit fully if back at/above reference
                    exit_ratio = 1.0
                exit_size = max(1, int(current_position * exit_ratio))
                exit_size = min(exit_size, current_position)
                exit_order_price = int(best_bid)
                self.sell(exit_order_price, exit_size)

        elif current_position < 0:  # Short position exit
            if best_ask < buy_exit_price:
                exit_ratio = self.params["partial_exit_ratio"]
                if best_ask <= reference_price:  # Exit fully if back at/below reference
                    exit_ratio = 1.0
                exit_size = max(1, int(abs(current_position) * exit_ratio))
                exit_size = min(exit_size, abs(current_position))
                exit_order_price = int(best_ask)
                self.buy(exit_order_price, exit_size)

        # Position scaling logic
        elif current_position > 0 and best_ask < (buy_entry_price - self.params["tier1_threshold"]):
            additional_deviation = (buy_entry_price - self.params["tier1_threshold"]) - best_ask
            additional_size = self._calculate_order_size(additional_deviation)
            available_capacity = position_limit - current_position
            qty_to_add = min(additional_size, available_capacity)

            if qty_to_add > 0:
                scale_order_price = int(best_ask)
                self.buy(scale_order_price, qty_to_add)

        elif current_position < 0 and best_bid > (sell_entry_price + self.params["tier1_threshold"]):
            additional_deviation = best_bid - (sell_entry_price + self.params["tier1_threshold"])
            additional_size = self._calculate_order_size(additional_deviation)
            available_capacity = position_limit + current_position
            qty_to_add = min(additional_size, available_capacity)

            if qty_to_add > 0:
                scale_order_price = int(best_bid)
                self.sell(scale_order_price, qty_to_add)

    def save(self) -> JSON:
        return {"mid_prices": self.mid_prices, "params": self.params, "ema_values": self.ema_values}

    def load(self, data: JSON) -> None:
        if data is not None:
            self.mid_prices = data.get("mid_prices", [])
            self.params = data.get("params", self.params)
            self.ema_values = data.get("ema_values", {})

class Trader:
    def __init__(self) -> None:
        limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
            "JAMS": 350,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
            "CROISSANTS": 250,
            "DJEMBES": 60,
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200,
            "MAGNIFICENT_MACARONS": 75,  # Add this line
        }

        self.strategies1: dict[Symbol, Strategy] = {symbol: clazz(symbol, limits[symbol]) for symbol, clazz in {
            "RAINFOREST_RESIN": RainforestResinStrategy,
            "KELP": KelpStrategy,
            "JAMS": JAMSStrategy,
            "PICNIC_BASKET1": BasketStrategy,
            "PICNIC_BASKET2": PicnicBasket2Strategy,
            "VOLCANIC_ROCK": Rock_MeanStrategy,
            "VOLCANIC_ROCK_VOUCHER_9500": MirrorVolcanicRockStrategy,
            "VOLCANIC_ROCK_VOUCHER_9750": MirrorVolcanicRockStrategy,
            "VOLCANIC_ROCK_VOUCHER_10000": MirrorVolcanicRockStrategy,
            "VOLCANIC_ROCK_VOUCHER_10250": MirrorVolcanicRockStrategy,
            "VOLCANIC_ROCK_VOUCHER_10500": MirrorVolcanicRockStrategy,
            "MAGNIFICENT_MACARONS": MagnificentMacaronsStrategy,  # Add this line
        }.items()}
        
        self.strategies2: dict[Symbol, Strategy] = {symbol: clazz(symbol, limits[symbol]) for symbol, clazz in {
            "RAINFOREST_RESIN": RainforestResinStrategy,
            "KELP": KelpStrategy,
            "JAMS": JAMSStrategy,
            "PICNIC_BASKET1": BasketStrategy,
            "PICNIC_BASKET2": PicnicBasket2Strategy,
            "VOLCANIC_ROCK": Rock_MeanStrategy,
            "VOLCANIC_ROCK_VOUCHER_9500": MirrorVolcanicRockStrategy,
            "VOLCANIC_ROCK_VOUCHER_9750": MirrorVolcanicRockStrategy,
            "VOLCANIC_ROCK_VOUCHER_10000": MirrorVolcanicRockStrategy,
            "VOLCANIC_ROCK_VOUCHER_10250": MirrorVolcanicRockStrategy,
            "VOLCANIC_ROCK_VOUCHER_10500": MaxLongPositionStrategy,
            "MAGNIFICENT_MACARONS": MagnificentMacaronsStrategy,  # Add this line
        }.items()}
        
        self.strategies3: dict[Symbol, Strategy] = {symbol: clazz(symbol, limits[symbol]) for symbol, clazz in {
            "RAINFOREST_RESIN": RainforestResinStrategy,
            "KELP": KelpStrategy,
            "JAMS": JAMSStrategy,
            "PICNIC_BASKET1": BasketStrategy,
            "PICNIC_BASKET2": PicnicBasket2Strategy,
            "VOLCANIC_ROCK": Rock_MeanStrategy,
            "VOLCANIC_ROCK_VOUCHER_9500": MirrorVolcanicRockStrategy,
            "VOLCANIC_ROCK_VOUCHER_9750": MirrorVolcanicRockStrategy,
            "VOLCANIC_ROCK_VOUCHER_10000": MirrorVolcanicRockStrategy,
            "VOLCANIC_ROCK_VOUCHER_10250": MaxLongPositionStrategy,
            "VOLCANIC_ROCK_VOUCHER_10500": MaxLongPositionStrategy,
            "MAGNIFICENT_MACARONS": MagnificentMacaronsStrategy,  # Add this line
        }.items()}
        
        self.strategies4: dict[Symbol, Strategy] = {symbol: clazz(symbol, limits[symbol]) for symbol, clazz in {
            "RAINFOREST_RESIN": RainforestResinStrategy,
            "KELP": KelpStrategy,
            "JAMS": JAMSStrategy,
            "PICNIC_BASKET1": BasketStrategy,
            "PICNIC_BASKET2": PicnicBasket2Strategy,
            "VOLCANIC_ROCK": Rock_MeanStrategy,
            "VOLCANIC_ROCK_VOUCHER_9500": MirrorVolcanicRockStrategy,
            "VOLCANIC_ROCK_VOUCHER_9750": MirrorVolcanicRockStrategy,
            "VOLCANIC_ROCK_VOUCHER_10000": MaxLongPositionStrategy,
            "VOLCANIC_ROCK_VOUCHER_10250": MaxLongPositionStrategy,
            "VOLCANIC_ROCK_VOUCHER_10500": MaxLongPositionStrategy,
            "MAGNIFICENT_MACARONS": MagnificentMacaronsStrategy,  # Add this line
        }.items()}
        
        self.strategies5: dict[Symbol, Strategy] = {symbol: clazz(symbol, limits[symbol]) for symbol, clazz in {
            "RAINFOREST_RESIN": RainforestResinStrategy,
            "KELP": KelpStrategy,
            "JAMS": JAMSStrategy,
            "PICNIC_BASKET1": BasketStrategy,
            "PICNIC_BASKET2": PicnicBasket2Strategy,
            "VOLCANIC_ROCK": Rock_MeanStrategy,
            "VOLCANIC_ROCK_VOUCHER_9500": MirrorVolcanicRockStrategy,
            "VOLCANIC_ROCK_VOUCHER_9750": MaxLongPositionStrategy,
            "VOLCANIC_ROCK_VOUCHER_10000": MaxLongPositionStrategy,
            "VOLCANIC_ROCK_VOUCHER_10250": MaxLongPositionStrategy,
            "VOLCANIC_ROCK_VOUCHER_10500": MaxLongPositionStrategy,
            "MAGNIFICENT_MACARONS": MagnificentMacaronsStrategy,  # Add this line
        }.items()}
                
        self.strategies = {}

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders = {}
        conversions = 0

        old_trader_data = json.loads(state.traderData) if state.traderData and state.traderData != "" else {}
        new_trader_data = {}
        all_external_orders = {}
        
        best_bid = max(state.order_depths["VOLCANIC_ROCK"].buy_orders.keys()) if "VOLCANIC_ROCK" in state.order_depths else 0
        best_ask = min(state.order_depths["VOLCANIC_ROCK"].sell_orders.keys()) if "VOLCANIC_ROCK" in state.order_depths else 0
        
        mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
        
        if mid_price > 0:
            self.strategies = self.strategies1 if mid_price > 10500 else self.strategies2 if mid_price > 10250 else self.strategies3 if mid_price > 10000 else self.strategies4 if mid_price > 9750 else self.strategies5
            
        for symbol, strategy in self.strategies.items():
            if symbol in old_trader_data:
                strategy.load(old_trader_data[symbol])

            if symbol in state.order_depths:
                strategy_orders, strategy_conversions = strategy.run(state)
                orders[symbol] = strategy_orders
                conversions += strategy_conversions
                
                # Collect any external orders from basket strategy
                if hasattr(strategy, 'external_orders') and strategy.external_orders:
                    for product, ext_orders in strategy.external_orders.items():
                        if product not in all_external_orders:
                            all_external_orders[product] = []
                        all_external_orders[product].extend(ext_orders)
                    # Clear external orders after processing
                    strategy.external_orders = {}

            new_trader_data[symbol] = strategy.save()
            
        # Add external orders to the result
        for product, ext_orders in all_external_orders.items():
            if product not in orders:
                orders[product] = []
            orders[product].extend(ext_orders)

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))
        
        logger.flush(
            state,
            orders,
            conversions,
            trader_data,
        )

        return orders, conversions, trader_data