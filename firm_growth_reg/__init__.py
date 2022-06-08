import pandas as pd
import numpy as np
import os
from init import rwrds, sedfile
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from scipy.stats.mstats import winsorize
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import string
from icecream import ic

pd.options.mode.use_inf_as_na = True
from latex_table.latex_table import latex
import warnings

warnings.simplefilter("ignore")


def ols(dataset, dep, ind, flag="i"):
    final = dict()
    Xs = [f"theta_{flag}_" + q for q in ind]
    Xs.extend(["capital", "employment", "sig2", "lnat"])
    exog = sm.add_constant(dataset[Xs])
    try:
        mod = PanelOLS(
            dataset[dep], exog, time_effects=True, other_effects=dataset["sic3"]
        )
        #     res = mod.fit(cov_type='clustered',cluster_entity=True,cluster_time=True)
        res = mod.fit(cov_type="clustered", cluster_entity=True)
    except:
        print(dep, ind)
        raise (ValueError)
    return res.params, res.tstats, res.nobs


def setIndex(df):
    return df.set_index(["gvkey", "fyear"])


# def ols(dataset, dep, ind, flag="i"):
def run_reg(df, Y, Xs, n, f):
    return ols(
        setIndex(df).dropna(subset=[Y + str(n), "capital", "employment", "sig2"]),
        Y + str(n),
        Xs,
        f,
    )


class reg:
    def __init__(
        self,
        df,
        Xs,
        year,
        des,
        name,
        end_t=11,
        Ys=["profits", "sale", "capital", "employment", "tfp"],
    ):
        assert len(df.groupby(["permno", "fyear"]).size()[lambda s: s > 1]) == 0
        self.df = df
        self.Xs = Xs
        self.year = year
        self.des = Path(des)
        self.name = name
        self.end_t = end_t
        self.Ys = Ys

        growth = rwrds("select * from tfp_compustat")
        growth = growth[
            ~(
                (growth.sic.isin(range(6000, 6800)))
                | (growth.sic.isin(range(4900, 4950)))
            )
        ]
        growth["cogs"] = growth["cogs"] / growth["sale"]
        growth["lnat"] = np.log(growth["at"] + 1)
        growth = growth[growth.fyear >= year]
        for c in [x for x in self.Ys if x != "tfp"]:
            growth.loc[growth[c].notnull(), c] = winsorize(
                growth.loc[growth[c].notnull(), c], limits=0.01
            )
        self.growth = growth[growth.fyear >= year]

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
        }

    def process_data(self):
        Xs = ["xi"] + self.Xs
        self.df = self.growth.merge(self.df, how="left")

        for X in Xs:
            self.df[X].fillna(0, inplace=True)
        self.df["sic3"] = self.df["sic"] // 10
        self.df = self.df.dropna(subset=["sic3"])
        # group_var = ["at"] + Xs
        # group_xi = self.df.groupby(["sic3", "fyear"])[group_var]
        #
        # sum_xi = (
        #     group_xi.sum()
        #     .reset_index()
        #     .merge(group_xi.size().reset_index(), on=["sic3", "fyear"])
        #     .rename({0: "numfirms"}, axis=1)[lambda df: df.numfirms > 1]
        # )
        # sum_xi.columns = (
        #     ["sic3", "fyear", "sum_at"] + [f"sum_{X}" for X in Xs] + ["numfirms"]
        # )
        # sum_xi = sum_xi[
        #     sum_xi["sic3"].isin(
        #         sum_xi.groupby("sic3")["sum_xi"].sum()[lambda s: s > 0].index
        #     )
        # ]
        # self.df = self.df.merge(sum_xi, on=["sic3", "fyear"])
        #
        for X in Xs:
            self.df[f"theta_i_{X}"] = self.df[X] / self.df["at"]
            # self.df[f"theta_j_{X}"] = (self.df[f"sum_{X}"] - self.df[X]) / (
            #     self.df.sum_at - self.df["at"]
            # )

        for c in [x for x in self.Ys if x != "tfp"]:
            self.df[c] = np.log(self.df[c]).fillna(0)

        self.df = self.df.groupby(["gvkey", "fyear"]).mean().reset_index()
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

    def estimate(self):
        all_tables = []
        self.process_data()

        _table_i, _table_j, b, t = latex(), latex(), dict(), dict()
        for panel, Y in zip(list(string.ascii_uppercase[: len(self.Ys)]), self.Ys):
            noobs = ["Obs."]
            for X in self.Xs:
                b[X], t[X], = (
                    [X],
                    [""],
                )

            with ProcessPoolExecutor(10) as p:
                result = []
                for time in range(1, self.end_t):
                    result.append(p.submit(run_reg, self.df, Y, self.Xs, time, "i"))
                for res in result:
                    params, tstats, nobs = res.result()
                    for X in self.Xs:
                        b[X].append(params.loc[f"theta_i_{X}"])
                        t[X].append(tstats[f"theta_i_{X}"])

                    noobs.append(nobs)
            _table_i.write_plain_row(
                "\multicolumn{%d}{c}{\\textit{Panel %s. %s}}"
                % (self.end_t * 2 - 2, panel, self.varname[Y])
            )
            _table_i.hline()
            for X in self.Xs:
                _table_i.collect_beta_t(zip(b[X], t[X]))
            _table_i.collect_row(noobs)
            _table_i.hline()
            _table_i.write_empty_row()

        the_table = self.des / f"firm_growth_i_{self.year}_{self.name}.tex"
        # the_table = des/f'firm_growth_i_{year}_level.tex'
        _table_i.write_table(the_table, "w")
        all_tables.append(the_table)

        # def run_reg(n):
        #     return self.ols(
        #         self.setIndex(self.df).dropna(
        #             subset=[Y + str(n), "capital", "employment", "sig2"]
        #         ),
        #         Y + str(n),
        #         self.Xs,
        #         f="j",
        #     )
        #
        # for panel, Y in zip(list(string.ascii_uppercase[: len(self.Ys)]), self.Ys):
        #     noobs = ["Obs."]
        #     for X in self.Xs:
        #         b[X], t[X], = (
        #             [X],
        #             [""],
        #         )
        #     with Pool() as p:
        #         for params, tstats, nobs in p.submit(run_reg, range(1, self.end_t)):
        #             for X in self.Xs:
        #                 b[X].append(params.loc[f"theta_j_{X}"])
        #                 t[X].append(tstats[f"theta_j_{X}"])
        #
        #             noobs.append(nobs)
        #     _table_j.write_plain_row(
        #         "\multicolumn{%d}{c}{\\textit{Panel %s. %s}}"
        #         % (end_t * 2 - 2, panel, varname[Y])
        #     )
        #     _table_j.hline()
        #     for X in self.Xs:
        #         _table_j.collect_beta_t(zip(b[X], t[X]))
        #     _table_j.collect_row(noobs)
        #     _table_j.hline()
        #
        # the_table = self.des / f"firm_growth_j_{self.year}_{self.name}.tex"
        # _table_j.write_table(the_table, "w")
        # all_tables.append(the_table)

        for t in all_tables:
            sedfile(r"s/_/\\_/g", t)
            sedfile(r"s/\\\\/\\/g", t)
            sedfile(r"s/\\$/\\\\/g", t)
            sedfile(r"s/\\ $/\\\\/g", t)

        for t in all_tables:
            os.system(f"fixtable -i --nostata {t} -v var.txt")
