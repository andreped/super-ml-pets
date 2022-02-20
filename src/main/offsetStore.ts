export interface IOffsetsStore {
	x64: IOffsets;
}

interface ISignature {
	sig: string;
	addressOffset: number;
	patternOffset: number;
}

export interface IOffsets {
	money: number[];
	hearts: number[];
	wins: number[];
	turns: number[];
	team0: {
		id: number[];
		attack: number[];
		health: number[];
	};
	team1: {
		id: number[];
		attack: number[];
		health: number[];
	};
	team2: {
		id: number[];
		attack: number[];
		health: number[];
	};
	team3: {
		id: number[];
		attack: number[];
		health: number[];
	};
    team4: {
		id: number[];
		attack: number[];
		health: number[];
	};
    shop0: {
		id: number[];
        frozen: number[];
		attack: number[];
		health: number[];
	};
	shop1: {
		id: number[];
        frozen: number[];
		attack: number[];
		health: number[];
	};
	shop2: {
		id: number[];
        frozen: number[];
		attack: number[];
		health: number[];
	};
	shop3: {
		id: number[];
        frozen: number[];
		attack: number[];
		health: number[];
	};
	shop4: {
		id: number[];
        frozen: number[];
		attack: number[];
		health: number[];
	};
	shop5: {
		id: number[];
        frozen: number[];
		attack: number[];
		health: number[];
	};
	shop6: {
		id: number[];
        frozen: number[];
		attack: number[];
		health: number[];
	};
	signatures: {
		innerNetClient: ISignature;
		meetingHud: ISignature;
		gameData: ISignature;
		shipStatus: ISignature;
	};
}

export default {
	x64: {
		money: [0x21d03e0, 0xb8, 0],
		hearts: [0x10],
		wins: [0xc0],
		turns: [0x21d0ea0, 0xb8, 0, 0xac],
		team0: {
			struct: [
				{ type: 'SKIP', skip: 16, name: 'unused' },
				{ type: 'UINT', name: 'id' },
				{ type: 'SKIP', skip: 4, name: 'unused' },
				{ type: 'UINT', name: 'name' },
				{ type: 'SKIP', skip: 4, name: 'unused' },
				{ type: 'UINT', name: 'color' },
				{ type: 'UINT', name: 'hat' },
				{ type: 'UINT', name: 'pet' },
				{ type: 'UINT', name: 'skin' },
				{ type: 'UINT', name: 'disconnected' },
				{ type: 'SKIP', skip: 4, name: 'unused' },
				{ type: 'UINT', name: 'taskPtr' },
				{ type: 'SKIP', skip: 4, name: 'unused' },
				{ type: 'BYTE', name: 'impostor' },
				{ type: 'BYTE', name: 'dead' },
				{ type: 'SKIP', skip: 6, name: 'unused' },
				{ type: 'UINT', name: 'objectPtr' },
				{ type: 'SKIP', skip: 4, name: 'unused' },
			],
			localX: [144, 108],
			localY: [144, 112],
			remoteX: [144, 88],
			remoteY: [144, 92],
			bufferLength: 80,
			offsets: [0, 0],
			inVent: [61],
			clientId: [40],
		},
		signatures: {
			innerNetClient: {
				sig:
					'48 8B 05 ? ? ? ? 48 8B 88 ? ? ? ? 48 8B 01 48 85 C0 0F 84 ? ? ? ? 66 66 66 0F 1F 84 00 ? ? ? ?',
				patternOffset: 3,
				addressOffset: 4,
			},
			meetingHud: {
				sig:
					'48 8B 05 ? ? ? ? 48 8B 88 ? ? ? ? 74 72 48 8B 39 48 8B 0D ? ? ? ? F6 81 ? ? ? ? ?',
				patternOffset: 3,
				addressOffset: 4,
			},
			gameData: {
				sig:
					'48 8B 05 ? ? ? ? 48 8B 88 ? ? ? ? 48 8B 01 48 85 C0 0F 84 ? ? ? ? BE ? ? ? ?',
				patternOffset: 3,
				addressOffset: 4,
			},
			shipStatus: {
				sig:
					'48 8B 05 ? ? ? ? 48 8B 5C 24 ? 48 8B 6C 24 ? 48 8B 74 24 ? 48 8B 88 ? ? ? ? 48 89 39 48 83 C4 20 5F',
				patternOffset: 3,
				addressOffset: 4,
			},
		},
	},

} as IOffsetsStore;
