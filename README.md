# Hymodel

Biblioteca Python para modelagem e simulação em recursos hídricos.

> **Status: em desenvolvimento (v0.1.0).** A API ainda pode mudar entre versões.

## Módulos

| Módulo | Conteúdo | Situação |
|---|---|---|
| `hymodel.hydrology` | Geração de séries sintéticas estocásticas (AR, ARMA, SARIMA) | Disponível |
| `hymodel.analysis` | Métricas de desempenho (RMSE, NSE, KGE, PBIAS) e gráficos | Disponível |
| `hymodel.common` | Constantes físicas, conversões de unidades e validação de entradas | Disponível |
| `hymodel.hydraulics` | Condutos, canais, vertedores | Planejado |
| `hymodel.hydrosedimentology` | Transporte de sedimentos, erosão, assoreamento | Planejado |

## Instalação

```bash
pip install hymodel
```

Para desenvolvimento, a partir do código-fonte:

```bash
git clone <url-do-repositorio>
cd Hymodel
pip install -e ".[dev]"
pytest
```

Requer Python 3.9 ou superior (dependências: numpy, scipy, pandas, matplotlib).

## Uso rápido

### Série sintética ARMA

```python
from hymodel.hydrology import Stoch

# 10 anos de vazão mensal sintética (m³/s), ARMA(1,1)
q = Stoch.arma(
    n_steps=120,
    mean=85.0,
    std_dev=20.0,
    ar_params=[0.6],
    ma_params=[0.3],
    seed=42,          # opcional: torna a série reprodutível
)
```

### Outros modelos

```python
# AR(2)
Stoch.ar_p(n_steps=500, mean=50.0, std_dev=10.0, ar_params=[0.5, 0.2])

# AR(1) com inovações assimétricas (skewness alvo = 1.2)
Stoch.ar_p_skewed(n_steps=500, mean=50.0, std_dev=10.0,
                  ar_params=[0.4], skewnesse=1.2)

# SARIMA(1,0,0)(1,0,0)_12 -- sazonalidade anual em dados mensais
Stoch.sarima(n_steps=240, mean=100.0, std_dev=15.0,
             ar_params=[0.3], ma_params=[],
             seasonal_ar_params=[0.6], seasonal_ma_params=[],
             seasonal_period=12)
```

Todos os métodos aceitam `init_value` (valores iniciais para continuar a partir de
dados observados) e `seed` (inteiro ou `numpy.random.Generator`).

### Métricas e gráficos

```python
from hymodel.analysis.metrics import nash_sutcliffe, kge, rmse, pbias
from hymodel.analysis.plots import plot_hidrograma

nse = nash_sutcliffe(q_obs, q_sim)
fig, ax = plot_hidrograma(t, q, titulo="Vazão sintética")
```

## Observações sobre os modelos estocásticos

- A média e o desvio padrão pedidos são impostos por reescalonamento empírico: o
  desvio sai exatamente igual ao solicitado e a média, aproximadamente.
- Os modelos são gaussianos (exceto `ar_p_skewed`); com média baixa e desvio alto
  podem surgir valores negativos. Para vazões, considere gerar em escala
  logarítmica e aplicar `numpy.exp` ao final.
- Em `ar_p_skewed`, a assimetria da série para AR(p) com p > 1 é aproximada
  (usa a soma dos coeficientes no lugar do coeficiente de correlação do AR(1)).
  Valide empiricamente com `scipy.stats.skew`.
- Os parâmetros (φ, θ) devem ser informados; a estimação a partir de séries
  observadas não faz parte da versão atual.

## Testes

```bash
pytest
```

## Referências

- Bras, R. L.; Rodriguez-Iturbe, I. *Random Functions and Hydrology*. Dover, 1985.
- Box, G. E. P.; Jenkins, G. M.; Reinsel, G. C. *Time Series Analysis: Forecasting and Control*. Wiley.

## Licença

Distribuído sob a licença MIT. Veja o arquivo [LICENSE](LICENSE).

## Autor

Jeferson Schreiner Junior