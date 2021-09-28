def profitable(r, ec, T, e_tax, cfix, cvar, demand, tf_at_peak):
    print((-(cfix + cvar) + sum([demand*(ec-e_tax) * 24 * tf_at_peak /( (1 + r)**n ) for n in range(1, T)])))
    return (-(cfix + cvar) + sum([demand*(ec-e_tax) * 24 * tf_at_peak /( (1 + r)**n ) for n in range(1, T)])) > 0

