from dataclasses import dataclass, field
import fileinput
import os
import re
import string
import sys
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import statsmodels.api as sm
from icecream import ic
from linearmodels.panel import PanelOLS
from loguru import logger
from scipy.stats.mstats import winsorize

import warnings

from .latex_table import latex

warnings.simplefilter("ignore")


def read_str(file, encoding="latin1"):
    with open(file, encoding=encoding) as f:
        return f.read().splitlines()


def star(p):
    p = abs(p)
    stars = ""
    if p < 0.1:
        stars = "*"
    if p <= 0.05:
        stars = "**"
    if p <= 0.01:
        stars = "***"
    return stars


def sedfile(string, filename):
    sta = string.split("/")
    if len(sta) == 4:
        _, input, output, oper = sta
    elif len(sta) == 3:
        _, input, oper = sta
    else:
        raise ValueError("cannot parse the input")

    if oper == "g":
        for line in fileinput.input(filename, inplace=1):
            if re.search(input, line):
                line = re.sub(input, output, line)
            sys.stdout.write(line)
    if oper == "d":
        for line in fileinput.input(filename, inplace=1):
            if re.search(input, line):
                continue
            sys.stdout.write(line)


def debug(s):
    logger.debug(s)


def setIndex(df):
    return df.set_index(["gvkey", "fyear"])


def ols(dataset, dep, ind, more_ctrl, flag="i", fe="industry"):
    final = dict()
    Xs = [f"theta_{flag}_" + q for q in ind]
    Xs.extend(["capital", "employment", "sig2", "lnat"] + more_ctrl)
    exog = sm.add_constant(dataset[Xs])
    if fe == "industry":
        try:
            mod = PanelOLS(
                dataset[dep], exog, time_effects=True, other_effects=dataset["sic3"]
            )
            #     res = mod.fit(cov_type='clustered',cluster_entity=True,cluster_time=True)
            res = mod.fit(cov_type="clustered", cluster_entity=True)
        except:
            print(dep, ind)
            raise (ValueError)

    elif fe == "firm":
        logger.info("estimating with Firm FE")
        try:
            mod = PanelOLS(dataset[dep], exog, time_effects=True, entity_effects=True)
            #     res = mod.fit(cov_type='clustered',cluster_entity=True,cluster_time=True)
            res = mod.fit(cov_type="clustered", cluster_entity=True)
        except:
            print(dep, ind)
            raise (ValueError)
    return mod, res


def run_reg(df, Y, Xs, more_ctrl, n, f, fe):
    return ols(
        setIndex(df).dropna(subset=[Y + str(n), "capital", "employment", "sig2"]),
        Y + str(n),
        Xs,
        more_ctrl,
        f,
        fe,
    )


@dataclass
class reg:
    df: pd.core.frame.DataFrame
    Xs: List
    growth_df: pd.core.frame.DataFrame
    year_start: int = 1975
    year_end: int = 2020
    des: str = None
    name: str = "output"
    more_ctrl: List[str] = field(default_factory=list)
    end_t: int = 11
    Ys: List[str] = field(
        default_factory=lambda: ["profits", "sale", "capital", "employment", "tfp"]
    )
    r2: bool = False
    include_fin_util: bool = False
    dont_scale_by_at: List[str] = field(default_factory=list)
    ctrl_y: bool = False
    ctrl_lag_y: bool = False
    ctrl_laglag_y: bool = False
    std_X: bool = False
    fe: str = "industry"
    wald: str = ""
    wald_diff: str = ""
    report_stat: str = "tvalues"
    report_stars: bool = True
    bdec: int = 3
    tdec: int = 2
    sdec: int = 3

    def preprocess(self):
        growth = self.growth_df
        assert (
            len(growth.groupby(["gvkey", "fyear"]).permno.count()[lambda s: s > 1]) == 0
        ), "not unique at gvkey,fyear"

        # check if columns are all presented
        growth[
            [
                "gvkey",
                "fyear",
                "permno",
                "sic",
                "at",
                "sale",
                "cogs",
                "invt",
                "ppegt",
                "employment",
                "capx",
                "profits",
                "output",
                "capital",
                "tfp",
                "sig2",
                "xrdat",
            ]
        ]

        for col in ["gvkey", "permno", "fyear"]:
            growth[col] = growth[col].astype(int)
        if not self.include_fin_util:
            growth = growth[
                ~(
                    (growth.sic.isin(range(6000, 6800)))
                    | (growth.sic.isin(range(4900, 4950)))
                )
            ]
        growth["cogs"] = growth["cogs"] / growth["sale"]
        growth["lnat"] = np.log(growth["at"] + 1)
        for c in [x for x in self.Ys if x != "tfp"]:
            growth.loc[growth[c].notnull(), c] = winsorize(
                growth.loc[growth[c].notnull(), c], limits=0.01
            )

        debug(f"Before sample period restriction, dataframe for est is {len(growth)}")
        self.growth = growth[lambda x: x.fyear >= self.year_start][
            lambda x: x.fyear <= self.year_end
        ]
        debug(
            f"After sample period restriction, dataframe for est is {len(self.growth)}"
        )

        self.varname = {
            "profits": "Profits",
            "output": "Output",
            "capital": "Capital",
            "employment": "Employment",
            "tfp": "Total Factor Productivity",
            "cogs": "Cost of Goods Sold scaled by Sale",
            "q": "Q",
            "sale": "Sales",
            "lerner": "Lerner Index",
            "at": "Asset",
        }

    def process_data(self):
        self.preprocess()
        # Xs = ["xi"] + self.Xs
        Xs = self.Xs
        self.df = self.growth.merge(self.df, how="left")
        debug(f"Merge with X, the df.len is {len(self.df)}")

        for X in Xs:
            self.df[X].fillna(0, inplace=True)
        self.df["sic3"] = self.df["sic"] // 10
        self.df = self.df.dropna(subset=["sic3"])
        debug(f"After drop missing SIC3, the df.len is {len(self.df)}")

        for X in Xs:
            if X in self.dont_scale_by_at:
                debug(f"not scaling {X} by assets")
                self.df[f"theta_i_{X}"] = self.df[X]
                continue
            self.df[f"theta_i_{X}"] = self.df[X] / self.df["at"]

        if self.std_X:
            for X in Xs:
                logger.info(f"standardizing theta_i_{X} to unit standard deviation")
                self.df[f"theta_i_{X}"] = (
                    self.df[f"theta_i_{X}"] / self.df[f"theta_i_{X}"].std()
                )

            # self.df[f"theta_j_{X}"] = (self.df[f"sum_{X}"] - self.df[X]) / (
            #     self.df.sum_at - self.df["at"]
            # )

        for c in [x for x in self.Ys if x != "tfp"]:
            self.df[c] = np.log(self.df[c]).fillna(0)

        self.df = self.df.groupby(["gvkey", "fyear"]).mean().reset_index()
        debug(f"After de-dup at gvkey, fyear, the df.len is {len(self.df)}")

        if self.ctrl_lag_y:
            debug("adding and controlling lagged Y")
            df = self.df[
                [
                    "gvkey",
                    "fyear",
                ]
                + self.Ys
            ]

            df = df.assign(fyear=df["fyear"] + 1)
            df.columns = ["gvkey", "fyear"] + list(
                map(lambda s: "theta_i_" + s + "l1", self.Ys)
            )
            self.df = self.df.merge(df, on=["gvkey", "fyear"])
            debug(f"dataframe after having lagged Y is {len(self.df)}")

            if self.ctrl_laglag_y:
                debug("adding and controlling lag lag Y")
                df = df.assign(fyear=df["fyear"] + 1)
                df.columns = [x.replace("l1", "l2") for x in df.columns]
                self.df = self.df.merge(df, on=["gvkey", "fyear"])
                debug(f"dataframe after having lag lag Y is {len(self.df)}")

        for tau in range(1, self.end_t):
            df = self.df[
                [
                    "gvkey",
                    "fyear",
                ]
                + self.Ys
            ]
            df = df.assign(fyear=df["fyear"] - tau)
            df.columns = ["gvkey", "fyear"] + list(map(lambda s: s + str(tau), self.Ys))
            self.df = self.df.merge(df, on=["gvkey", "fyear"], how="left")

            # levels versus change
            for c, d in zip(self.Ys, list(map(lambda s: s + str(tau), self.Ys))):
                self.df[d] = self.df[d] - self.df[c]

        for Y in self.Ys:
            self.df["theta_i_" + Y] = self.df[Y]

        self.df = self.df.replace([np.inf, -np.inf], np.nan)

    def estimate(self):
        all_tables = []
        self.process_data()

        logger.info("estimating with more controls {}".format(self.more_ctrl))

        _table_i, _table_j, b, stats = (
            latex(bdec=self.bdec, tdec=self.tdec, sdec=self.sdec),
            latex(),
            dict(),
            dict(),
        )
        for panel, Y in zip(list(string.ascii_uppercase[: len(self.Ys)]), self.Ys):
            ctrl_y = []
            if self.ctrl_y and Y not in ["employment", "capital"]:
                ctrl_y.append(Y)
            if self.ctrl_lag_y:
                ctrl_y.append(f"{Y}l1")
            if self.ctrl_laglag_y:
                ctrl_y.append(f"{Y}l2")
            noobs = ["Obs."]
            r2s = [r"$R^2 (\%)$"]
            wald_test = ["Wald"]
            wald_diff = [self.wald_diff] if self.wald_diff else ["Diff"]
            for X in self.Xs:
                (
                    b[X],
                    stats[X],
                ) = (
                    [X],
                    [""],
                )

            for time in range(1, self.end_t):
                model, res = run_reg(
                    self.df,
                    Y,
                    ctrl_y + self.Xs,
                    self.more_ctrl,
                    time,
                    "i",
                    self.fe,
                )
                for X in self.Xs:
                    b[X].append(res.params.loc[f"theta_i_{X}"])
                    if self.report_stat == "tvalues":
                        stats[X].append(res.tstats[f"theta_i_{X}"])
                    elif self.report_stat == "bse":
                        stats[X].append(res.std_errors[f"theta_i_{X}"])

                noobs.append(res.nobs)
                r2s.append(res.rsquared * 100)
                if self.wald:
                    wald = res.wald_test(formula=self.wald)
                    v1, v2 = self.wald.split("=")
                    wald_diff.append(
                        str(round(res.params.loc[v1] - res.params.loc[v2], 3))
                        + star(wald.pval)
                    )
                    wald_test.append(str(round(wald.stat, 3)))
            _table_i.write_plain_row(
                "\\multicolumn{%d}{c}{\\textit{Panel %s. %s}}"
                % (self.end_t * 2 - 2, panel, self.varname[Y])
            )
            _table_i.hline()
            for X in self.Xs:
                _table_i.collect_beta_t(
                    zip(b[X], stats[X]),
                    se=self.report_stat == "bse",
                    stars=self.report_stars,
                )
            if self.r2:
                _table_i.collect_row(r2s, rounding=2)
            if self.wald:
                _table_i.write_empty_row()
                _table_i.collect_row(wald_diff)
                _table_i.collect_row(wald_test)

            _table_i.write_empty_row()
            _table_i.collect_row(noobs)
            _table_i.hline()
            _table_i.write_empty_row()

        the_table = Path(self.des) / f"firm_growth_i_{self.year_start}_{self.name}.tex"
        self.table_name = the_table.as_posix()
        _table_i.rows = _table_i.rows[:-1]
        logger.info(f"writing table {self.table_name}")
        _table_i.write_table(the_table, "w")
        all_tables.append(the_table)

        for t in all_tables:
            sedfile(r"s/_/\\_/g", t)
            sedfile(r"s/\\\\/\\/g", t)
            sedfile(r"s/\\$/\\\\/g", t)
            sedfile(r"s/\\ $/\\\\/g", t)

    def plotter(
        self,
        ax: plt.Axes,
        var: List[float],
        var_t: List[float],
        legend: str,
        color: str,
        error_bars: bool,
    ) -> plt.Axes:
        time_intervals = list(range(1, self.end_t))

        estimates = list(map(float, var))
        std_errors = [
            est / abs(float(t_val) + 0.01) for est, t_val in zip(estimates, var_t)
        ]
        lower_bounds = [est - 1.96 * se for est, se in zip(estimates, std_errors)]
        upper_bounds = [est + 1.96 * se for est, se in zip(estimates, std_errors)]

        ax.plot(
            time_intervals,
            estimates,
            marker="o",
            linestyle="-",
            label=legend,
            color=color,
        )
        if error_bars:
            ax.plot(
                time_intervals,
                lower_bounds,
                linestyle="--",
                color=color,
                alpha=0.6,
            )
            ax.plot(
                time_intervals,
                upper_bounds,
                linestyle="--",
                color=color,
                alpha=0.6,
            )
        # ax.fill_between(
        #     time_intervals, lower_bounds, upper_bounds, color=color, alpha=0.2
        # )

        ax.set_xticks(time_intervals)
        ax.set_xticklabels([f"t+{i}" for i in time_intervals])

        ax.set_xlabel("")
        ax.set_ylabel("Beta")
        ax.legend(facecolor="white")
        ax.set_facecolor("white")
        ax.grid(True, color="lightgrey")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    def plot_beta(
        self,
        exclude: List[str] = [],
        error_bars: bool = True,
        colors=[
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
            "tab:olive",
            "tab:cyan",
        ],
    ):
        tb = read_str(self.table_name)
        varname, coef, tstat = [], [], []

        for i, x in enumerate(tb):
            if x.startswith("&"):
                tstat.append(re.findall(r"\d\.\d+", x))
                row = tb[i - 1]
                varname.append(row.split("&")[0].strip())
                coef.append(re.findall(r"\d\.\d+", row))

        N = 0
        fig, ax = plt.subplots(figsize=(8, 6))
        for v, co, t, color in zip(varname, coef, tstat, colors[: len(self.Xs)] * 10):
            print(v)
            N += 1
            if not any(ex.lower() in v.lower() for ex in exclude):
                self.plotter(ax, co, t, v, color, error_bars)

            if N % len(self.Xs) == 0:
                outfile = f"{self.table_name.split('.tex')[0]}_{self.Ys[(N // len(self.Xs))-1]}.png"
                # ax.title.set_text(f"{self.Ys[(N // len(self.Xs))-1]}")
                fig.savefig(outfile, bbox_inches="tight")
                print(f"Saved plot to {outfile}")
                fig, ax = plt.subplots(figsize=(8, 6))
