from consulta_jurisprudencia import ConsultaJurisprudencia

#  Inicializa a consulta
consulta = ConsultaJurisprudencia()

#  Consulta simples para teste
resposta = consulta.consultar("Qual o entendimento sobre o uso de Canabidiol em Goiás?")
print("\n Resposta do Modelo:")
print(resposta)
