CREATE TABLE daily_tech (
                symbol   TEXT NOT NULL,
                date    TEXT NOT NULL,
                open    REAL,
                high    REAL,
                low     REAL,
                close   REAL,
                volume  INTEGER,
                PRIMARY KEY (symbol, date)
            );
CREATE TABLE daily_indx (
                symbol   TEXT NOT NULL,
                date    TEXT NOT NULL,
                open    REAL,
                high    REAL,
                low     REAL,
                close   REAL,
                PRIMARY KEY (symbol, date)
            );
CREATE TABLE intra_spy (
                date_time   TEXT,
                open        REAL,
                high        REAL,
                low         REAL,
                close       REAL,
                volume      INTEGER,
                bar_count   INTEGER,
                avg_price   REAL,
                PRIMARY KEY (date_time)
            );
CREATE TABLE mkt_indx (
                date    TEXT NOT NULL,
                vwretd  REAL,
                vwretx  REAL,
                ewretd  REAL,
                sprtrn  REAL,
                spindx  REAL,
                totval  REAL,
                usdval  REAL,
                PRIMARY KEY (date)
            );
CREATE TABLE intra_misc (
                date_time   TEXT NOT NULL,
                es          REAL,
                nq          REAL,
                rty         REAL,
                spy         REAL,
                qqq         REAL,
                iwm         REAL,
                aapl        REAL,
                msft        REAL,
                nvda        REAL,
                xlk         REAL,
                xlf         REAL,
                xlp         REAL,
                xly         REAL,
                xtn         REAL,
                hyg         REAL,
                tnx         REAL,
                tyx         REAL,
                es_volume   INTEGER,
                tlt         REAL,
                tlt_volume  INTEGER,
                vix         REAL,
                spx         REAL,
                spx_pcr     REAL,
                spy_pcr     REAL,
                es_pcr      REAL,
                PRIMARY KEY (date_time)
            );
